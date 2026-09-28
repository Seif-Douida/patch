-- One row per day, 2020-01-01 to 2030-12-31, flagged with Steam's seasonal sales.
with digits as (
    select d.n from (values (0), (1), (2), (3), (4), (5), (6), (7), (8), (9)) as d (n)
),

numbers as (
    select a.n + 10 * b.n + 100 * c.n + 1000 * d.n as n
    from digits as a
    cross join digits as b
    cross join digits as c
    cross join digits as d
),

days as (
    select dateadd(day, n, cast('2020-01-01' as date)) as date_day
    from numbers
    where n <= datediff(day, '2020-01-01', '2030-12-31')
)

select
    cast(convert(char(8), days.date_day, 112) as int) as date_key,
    days.date_day as [date],
    -- 1 = Monday regardless of the server's DATEFIRST (1900-01-01 was a Monday).
    datediff(day, cast('1900-01-01' as date), days.date_day) % 7 + 1 as dow,
    datepart(iso_week, days.date_day) as iso_week,
    month(days.date_day) as [month],
    year(days.date_day) as [year],
    cast(case when sales.sale_name is null then 0 else 1 end as bit) as is_steam_sale,
    sales.sale_name
from days
left join {{ ref('steam_sale_calendar') }} as sales
    on days.date_day between sales.start_date and sales.end_date
