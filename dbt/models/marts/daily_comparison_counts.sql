{{ config(materialized='table') }}
-- One row per comparison game (role donor) and UTC day: its positive and negative reviews, from
-- Steam's per-day counts (Phase 4, design D4). The causal module's synthetic control builds its
-- twins from these; tracked games' days come from their own reviews (daily_game_sentiment).
select
    g.game_key,
    cast(convert(char(8), c.day, 112) as int) as date_key,
    c.positive as n_positive,
    c.negative as n_negative,
    c.positive + c.negative as n_reviews
from {{ ref('stg_steam_review_daily') }} as c
inner join {{ ref('dim_game') }} as g
    on g.appid = c.appid and g.role = 'donor'
