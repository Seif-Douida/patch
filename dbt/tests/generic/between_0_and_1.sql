{#- A share or probability must lie in [0, 1]; NULL (no data) is allowed. -#}
{% test between_0_and_1(model, column_name) %}
select *
from {{ model }}
where {{ column_name }} < 0 or {{ column_name }} > 1
{% endtest %}
