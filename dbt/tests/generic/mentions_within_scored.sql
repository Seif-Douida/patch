{#- An aspect can't be mentioned by more reviews than were scored (mart.daily_aspect_share). -#}
{% test mentions_within_scored(model) %}
select *
from {{ model }}
where n_neg_mentions > n_neg_scored or n_pos_mentions > n_pos_scored
{% endtest %}
