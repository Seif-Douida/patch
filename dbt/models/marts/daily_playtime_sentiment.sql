{{ config(materialized='table') }}
-- One row per game, day and playtime bucket at review: its reviews and positive reviews. Phase 4's
-- decompose_change splits a change in positive share into who reviewed and how they felt (design
-- D8). Reviews whose playtime Steam didn't send are 'unknown'. Rebuilt each night: a few seconds
-- over every review.
select
    f.game_key,
    f.date_key,
    coalesce(f.playtime_bucket, 'unknown') as playtime_bucket,
    count(*) as n_reviews,
    sum(cast(f.voted_up as int)) as n_positive
from {{ ref('fact_review') }} as f
group by f.game_key, f.date_key, coalesce(f.playtime_bucket, 'unknown')
