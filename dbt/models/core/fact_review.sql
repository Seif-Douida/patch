{{ config(
    materialized='incremental',
    unique_key='review_id',
    incremental_strategy='merge',
    merge_exclude_columns=['is_embedded'],
    full_refresh=false
) }}

-- One row per review: its latest version. Each run only touches reviews the pipeline saw since
-- the previous build (new reviews, new versions, refreshed vote counts).
with changed as (
    select distinct recommendation_id
    from {{ source('raw', 'steam_review') }}
    {% if is_incremental() %}
        where last_seen_run_id > (select coalesce(max(last_seen_run_id), 0) from {{ this }})
    {% endif %}
)

select
    r.review_id,
    r.appid as game_key,
    cast(convert(char(8), r.timestamp_created, 112) as int) as date_key,
    r.language as language_key,
    r.author_hash,
    r.voted_up,
    r.author_playtime_at_review as playtime_at_review_min,
    case
        when r.author_playtime_at_review is null then null
        when r.author_playtime_at_review < 120 then 'lt_2h'
        when r.author_playtime_at_review < 600 then '2h_10h'
        when r.author_playtime_at_review < 3000 then '10h_50h'
        when r.author_playtime_at_review < 12000 then '50h_200h'
        else '200h_plus'
    end as playtime_bucket,
    r.votes_up,
    r.weighted_vote_score,
    r.steam_purchase,
    r.received_for_free,
    r.written_during_early_access as early_access,
    {{ word_count('r.review_text') }} as text_len_words,
    -- Set by Phase 3's embedding job; merges never overwrite it.
    cast(0 as bit) as is_embedded,
    r.first_seen_run_id,
    r.last_seen_run_id,
    r.timestamp_created as created_at,
    r.timestamp_updated as updated_at
from {{ ref('stg_steam_review') }} as r
inner join changed as c on c.recommendation_id = r.review_id
