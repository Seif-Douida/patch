-- The reviews Qdrant holds (Phase 3, design D1), with the fields search filters on (design D2),
-- the review's version and a fingerprint of the fields. The nightly index stage compares this
-- table with ops.qdrant_point, its record of what Qdrant holds (design D5): a new version is
-- embedded again, a new fingerprint only updates the fields, and a review that left is deleted.
--
-- A table rebuilt each night rather than incremental: the per-game cap moves membership whenever
-- new reviews arrive, so an incremental model would need its own deletions.
--
-- Tracked games only (role tracked or both): a donor game feeds the marts and the causal module,
-- and would take up to its cap of the free cluster's memory for searches nobody runs.
{%- set char_languages = "'" ~ var('index_char_languages') | join("', '") ~ "'" %}
{%- set window = var('index_patch_window_days') %}

with eligible as (
    select
        f.review_id,
        f.game_key,
        f.date_key,
        f.created_at,
        f.updated_at,
        f.language_key,
        f.voted_up,
        f.playtime_bucket,
        f.first_seen_run_id
    from {{ ref('fact_review') }} as f
    inner join {{ ref('dim_game') }} as g
        on g.game_key = f.game_key and g.role in ('tracked', 'both')
    left join {{ ref('review_text') }} as t
        on t.review_id = f.review_id and f.language_key in ({{ char_languages }})
    where
        (f.language_key in ({{ char_languages }}) and len(t.[text]) >= {{ var('index_min_chars') }})
        or (
            f.language_key not in ({{ char_languages }})
            and f.text_len_words >= {{ var('index_min_words') }}
        )
),

-- The days the windows are centred on: each treatment-candidate patch, and the game's launch,
-- when the reviews that shaped its reputation were written.
window_days as (
    select
        game_key,
        cast(published_at as date) as anchor_day
    from {{ ref('fact_patch') }}
    where is_treatment_candidate = 1
    union
    select
        game_key,
        release_date
    from {{ ref('dim_game') }}
    where release_date is not null
),

ranked as (
    select
        e.*,
        row_number() over (
            partition by e.game_key order by e.created_at desc, e.review_id desc
        ) as recency_rank
    from eligible as e
),

kept as (
    select r.*
    from ranked as r
    where
        r.recency_rank <= {{ var('index_cap_per_game') }}
        or exists (
            select 1
            from window_days as w
            where
                w.game_key = r.game_key
                and cast(r.created_at as date)
                between dateadd(day, -{{ window }}, w.anchor_day)
                and dateadd(day, {{ window }}, w.anchor_day)
        )
),

-- The positive labels of each review's latest score, in taxonomy order.
aspects as (
    select
        review_id,
        string_agg(cast(aspect_key as nvarchar(max)), '|') within group (
            order by case aspect_key
                {%- for code in var('aspects') %}
                when '{{ code }}' then {{ loop.index }}
                {%- endfor %}
            end
        ) as aspects
    from {{ ref('bridge_review_aspect') }}
    group by review_id
),

fields as (
    select
        k.review_id,
        k.game_key as appid,
        k.date_key as [day],
        datediff_big(second, '1970-01-01', k.created_at) as created_ts,
        cast(k.language_key as nvarchar(32)) as [language],
        k.voted_up,
        cast(k.playtime_bucket as nvarchar(16)) as playtime_bucket,
        k.first_seen_run_id,
        k.updated_at as [version],
        cast(coalesce(a.aspects, '') as nvarchar(400)) as aspects,
        -- The latest score's model, by the scoring stage's own rule, so a review scored with no
        -- positive label still names the model that said so; null while unscored.
        cast(s.model_version as nvarchar(64)) as aspect_model_version,
        cast('{{ var("index_rules_version") }}' as nvarchar(8)) as rules_version
    from kept as k
    left join aspects as a on a.review_id = k.review_id
    outer apply (
        select top 1 sc.model_version
        from {{ source('raw', 'review_aspect_score') }} as sc
        where sc.recommendation_id = k.review_id and sc.timestamp_updated = k.updated_at
        order by sc.scored_run_id desc, sc.scored_at desc, sc.model_version desc
    ) as s
)

-- The fingerprint covers every field but the version (null fields count as empty).
select
    f.*,
    cast(lower(convert(char(64), hashbytes('SHA2_256', concat(
        f.appid, '|', f.[day], '|', f.created_ts, '|', f.[language], '|', f.voted_up, '|',
        f.playtime_bucket, '|', f.first_seen_run_id, '|', f.aspects, '|',
        f.aspect_model_version, '|', f.rules_version
    )), 2)) as char(64)) as fingerprint
from fields as f
