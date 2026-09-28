{#-
    Flags a game-day whose review count is more than `z` standard deviations from the mean of the
    previous `window` days (spec §6.4), once at least `min_history` days of history exist.
    A warning, not an error: review bombs and launches are real events worth a look.
-#}
{% test daily_volume_anomaly(model, column_name, z=4, window=28, min_history=14) %}
{{ config(severity='warn') }}
with history as (
    select
        game_key,
        date_key,
        {{ column_name }} as volume,
        avg(cast({{ column_name }} as float)) over (
            partition by game_key order by date_key rows between {{ window }} preceding and 1 preceding
        ) as mean_before,
        stdev(cast({{ column_name }} as float)) over (
            partition by game_key order by date_key rows between {{ window }} preceding and 1 preceding
        ) as sd_before,
        count(*) over (
            partition by game_key order by date_key rows between {{ window }} preceding and 1 preceding
        ) as days_before
    from {{ model }}
)

select *
from history
where days_before >= {{ min_history }}
    and sd_before > 0
    and abs(volume - mean_before) / sd_before > {{ z }}
{% endtest %}
