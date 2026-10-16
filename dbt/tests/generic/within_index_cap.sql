{#- Fails for any game whose index holds more than `cap` reviews outside the `window_days` either
    side of its treatment-candidate patches and its launch (design D1). -#}
{% test within_index_cap(model, cap, window_days) %}
with anchors as (
    select game_key, cast(published_at as date) as anchor_day
    from {{ ref('fact_patch') }}
    where is_treatment_candidate = 1
    union
    select game_key, release_date
    from {{ ref('dim_game') }}
    where release_date is not null
)

select i.appid, count(*) as n
from {{ model }} as i
where not exists (
    select 1
    from anchors as a
    where
        a.game_key = i.appid
        and datefromparts(i.[day] / 10000, i.[day] / 100 % 100, i.[day] % 100)
        between dateadd(day, -{{ window_days }}, a.anchor_day)
        and dateadd(day, {{ window_days }}, a.anchor_day)
)
group by i.appid
having count(*) > {{ cap }}
{% endtest %}
