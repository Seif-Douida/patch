{#- Fails for any game whose index holds more than `cap` reviews outside the `window_days` either
    side of its treatment-candidate patches (design D1). -#}
{% test within_index_cap(model, cap, window_days) %}
select i.appid, count(*) as n
from {{ model }} as i
where not exists (
    select 1
    from {{ ref('fact_patch') }} as p
    where
        p.game_key = i.appid
        and p.is_treatment_candidate = 1
        and datefromparts(i.[day] / 10000, i.[day] / 100 % 100, i.[day] % 100)
        between dateadd(day, -{{ window_days }}, cast(p.published_at as date))
        and dateadd(day, {{ window_days }}, cast(p.published_at as date))
)
group by i.appid
having count(*) > {{ cap }}
{% endtest %}
