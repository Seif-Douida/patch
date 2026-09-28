{{ config(
    materialized='incremental',
    unique_key=['game_key', 'date_key'],
    incremental_strategy='delete+insert'
) }}

-- One row per game per day, from the game's first stored review to today. Days without reviews
-- have n_reviews = 0 and NULL shares. Incremental runs rebuild only the affected days: days with
-- reviews seen since the previous build, days older than the game's previous first day (history
-- the backfill just added), and the last 4 days (new empty days, today's price).

with clock as (
    select cast(sysutcdatetime() as date) as today
),

first_day as (
    select game_key, min(date_key) as first_date_key
    from {{ ref('fact_review') }}
    group by game_key
),

spine as (
    select g.game_key, g.appid, d.date_key, d.[date], d.is_steam_sale
    from {{ ref('dim_game') }} as g
    inner join first_day as f on f.game_key = g.game_key
    inner join {{ ref('dim_date') }} as d
        on d.date_key >= f.first_date_key and d.[date] <= (select today from clock)
),

{% if is_incremental() %}
previous_build as (
    select coalesce(max(source_run_id), 0) as run_id from {{ this }}
),

previous_first_day as (
    select game_key, min(date_key) as first_date_key from {{ this }} group by game_key
),

affected as (
    select distinct game_key, date_key
    from {{ ref('fact_review') }}
    where last_seen_run_id > (select run_id from previous_build)
    union
    select s.game_key, s.date_key
    from spine as s
    left join previous_first_day as p on p.game_key = s.game_key
    where
        p.first_date_key is null
        or s.date_key < p.first_date_key
        or s.[date] >= dateadd(day, -4, (select today from clock))
),

days as (
    select s.*
    from spine as s
    inner join affected as a on a.game_key = s.game_key and a.date_key = s.date_key
),
{% else %}
days as (
    select * from spine
),
{% endif %}

reviews as (
    select f.game_key, f.date_key, f.voted_up, f.playtime_at_review_min
    from {{ ref('fact_review') }} as f
    inner join days as d on d.game_key = f.game_key and d.date_key = f.date_key
),

counts as (
    select
        game_key,
        date_key,
        count(*) as n_reviews,
        sum(cast(voted_up as int)) as n_positive,
        sum(case when playtime_at_review_min < 120 then 1 else 0 end) as n_new_players
    from reviews
    group by game_key, date_key
),

medians as (
    select distinct
        game_key,
        date_key,
        percentile_cont(0.5) within group (order by playtime_at_review_min)
            over (partition by game_key, date_key) as median_playtime
    from reviews
),

prices as (
    select game_key, date_key, discount_pct
    from {{ ref('fact_price_snapshot') }}
    where ok = 1
)

select
    d.game_key,
    d.appid,
    d.date_key,
    d.[date],
    coalesce(c.n_reviews, 0) as n_reviews,
    coalesce(c.n_positive, 0) as n_positive,
    case when c.n_reviews > 0 then cast(c.n_positive as float) / c.n_reviews end as pos_share,
    {{ wilson_lower('c.n_positive', 'c.n_reviews') }} as pos_lo90,
    {{ wilson_upper('c.n_positive', 'c.n_reviews') }} as pos_hi90,
    m.median_playtime as median_playtime_at_review,
    case
        when c.n_reviews > 0 then cast(c.n_new_players as float) / c.n_reviews
    end as share_new_players,
    p.discount_pct,
    d.is_steam_sale as is_sale,
    (select coalesce(max(last_seen_run_id), 0) from {{ ref('fact_review') }}) as source_run_id
from days as d
left join counts as c on c.game_key = d.game_key and c.date_key = d.date_key
left join medians as m on m.game_key = d.game_key and m.date_key = d.date_key
left join prices as p on p.game_key = d.game_key and p.date_key = d.date_key
