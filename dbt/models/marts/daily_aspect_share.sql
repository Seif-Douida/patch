{{ config(
    materialized='incremental',
    unique_key=['game_key', 'date_key', 'aspect_key'],
    incremental_strategy='delete+insert'
) }}

-- Per game, day and aspect (design D8): how many of the day's negative and positive reviews
-- mention the aspect, out of how many of them are scored. A review counts as scored once its
-- latest version has a score, so partial scoring shows as n_neg_scored + n_pos_scored below the
-- day's review count, and the shares are NULL when nothing that day is scored. Days with reviews
-- get one row per aspect. Incremental runs rebuild only the days of the reviews that changed since
-- the previous build (new versions or new scores).

with
{% if is_incremental() %}
affected as (
    select distinct f.game_key, f.date_key
    from {{ ref('fact_review') }} as f
    where f.review_id in (
        {{ aspect_changed_reviews('(select coalesce(max(source_run_id), 0) from ' ~ this ~ ')') }}
    )
),
{% endif %}

reviews as (
    select
        f.review_id,
        f.game_key,
        f.date_key,
        f.voted_up,
        case
            when exists (
                select 1
                from {{ source('raw', 'review_aspect_score') }} as s
                where s.recommendation_id = f.review_id and s.timestamp_updated = f.updated_at
            ) then 1
            else 0
        end as is_scored
    from {{ ref('fact_review') }} as f
    {% if is_incremental() %}
        inner join affected as a on a.game_key = f.game_key and a.date_key = f.date_key
    {% endif %}
),

scored as (
    select
        game_key,
        date_key,
        sum(case when voted_up = 0 then is_scored else 0 end) as n_neg_scored,
        sum(case when voted_up = 1 then is_scored else 0 end) as n_pos_scored
    from reviews
    group by game_key, date_key
),

mentions as (
    select
        r.game_key,
        r.date_key,
        b.aspect_key,
        sum(case when r.voted_up = 0 then 1 else 0 end) as n_neg_mentions,
        sum(case when r.voted_up = 1 then 1 else 0 end) as n_pos_mentions
    from reviews as r
    inner join {{ ref('bridge_review_aspect') }} as b on b.review_id = r.review_id
    group by r.game_key, r.date_key, b.aspect_key
)

select
    s.game_key,
    s.date_key,
    a.aspect_key,
    coalesce(m.n_neg_mentions, 0) as n_neg_mentions,
    coalesce(m.n_pos_mentions, 0) as n_pos_mentions,
    s.n_neg_scored,
    s.n_pos_scored,
    case
        when s.n_neg_scored > 0 then cast(coalesce(m.n_neg_mentions, 0) as float) / s.n_neg_scored
    end as share_of_negative,
    case
        when s.n_pos_scored > 0 then cast(coalesce(m.n_pos_mentions, 0) as float) / s.n_pos_scored
    end as share_of_positive,
    {{ aspect_source_run_id() }} as source_run_id
from scored as s
cross join {{ ref('dim_aspect') }} as a
left join mentions as m
    on m.game_key = s.game_key and m.date_key = s.date_key and m.aspect_key = a.aspect_key
