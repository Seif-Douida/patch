{#- Approximate word count: spaces + 1 after turning line breaks into spaces. Scripts without
    spaces (Chinese, Japanese) count as one word; Phase 3 refines this for Qdrant's 20-word rule. -#}
{% macro word_count(text) -%}
    case when len(ltrim(rtrim({{ text }}))) = 0 then 0 else
        len(ltrim(rtrim(replace(replace({{ text }}, char(13), ' '), char(10), ' '))))
        - len(replace(ltrim(rtrim(replace(replace({{ text }}, char(13), ' '), char(10), ' '))), ' ', ''))
        + 1
    end
{%- endmacro %}
