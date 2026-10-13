{#-
    The aspect models' incremental runs (design D8). A review's aspects change when it gets a new
    version (core.fact_review) or a new score (raw.review_aspect_score): both carry the pipeline
    run that wrote them, and each build stamps its rows with the latest such run.
-#}

{#- The reviews whose aspects changed after `run_id` (a SQL expression). -#}
{% macro aspect_changed_reviews(run_id) %}
    select review_id
    from {{ ref('fact_review') }}
    where last_seen_run_id > {{ run_id }}
    union
    select recommendation_id
    from {{ source('raw', 'review_aspect_score') }}
    where scored_run_id > {{ run_id }}
{% endmacro %}

{#- The latest run whose reviews or scores this build has read. -#}
{% macro aspect_source_run_id() %}
    (
        select coalesce(max(run_id), 0)
        from (
            select max(last_seen_run_id) as run_id from {{ ref('fact_review') }}
            union all
            select max(scored_run_id) from {{ source('raw', 'review_aspect_score') }}
        ) as runs
    )
{% endmacro %}
