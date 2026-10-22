-- Comparison games' review counts per UTC day (Phase 4, design D4): the history fetched once (the
-- seed donor_daily_counts) and the days the nightly counts stage has stored since. A day in both
-- takes the stored row: it was fetched later.
with stored as (
    select appid, day, positive, negative
    from {{ source('raw', 'steam_review_daily') }}
),

seeded as (
    select s.appid, s.day, s.positive, s.negative
    from {{ ref('donor_daily_counts') }} as s
    where not exists (
        select 1 from stored where stored.appid = s.appid and stored.day = s.day
    )
)

select appid, day, positive, negative from stored
union all
select appid, day, positive, negative from seeded
