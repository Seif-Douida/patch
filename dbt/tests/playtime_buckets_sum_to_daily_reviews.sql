-- Every game-day's playtime buckets add up to its reviews in daily_game_sentiment (Phase 4, D8).
select s.game_key, s.date_key, s.n_reviews, coalesce(p.n_reviews, 0) as bucket_reviews
from {{ ref('daily_game_sentiment') }} as s
left join (
    select game_key, date_key, sum(n_reviews) as n_reviews
    from {{ ref('daily_playtime_sentiment') }}
    group by game_key, date_key
) as p on p.game_key = s.game_key and p.date_key = s.date_key
where coalesce(p.n_reviews, 0) <> s.n_reviews
