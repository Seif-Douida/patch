{{ config(
    materialized='incremental',
    unique_key='review_id',
    incremental_strategy='delete+insert',
    post_hook="delete from {{ this }} where aspect_key is null"
) }}

-- The positive labels of the latest score of each review's latest version (design D8): one row
-- per review and aspect it mentions. A review whose latest version isn't scored yet has no rows.
-- Rebuildable from raw.review_aspect_score, so a full refresh is allowed.
--
-- Incremental runs replace the rows of the reviews that changed since the previous build (new
-- versions, new scores, a new champion): delete+insert by review_id. A changed review left with
-- no positive label comes out of the left join as one placeholder row (aspect_key NULL), so its
-- old rows are deleted too; the post-hook removes the placeholders in the same transaction.

with latest as (
    select f.review_id, f.updated_at
    from {{ ref('fact_review') }} as f
    {% if is_incremental() %}
        where f.review_id in (
            {{ aspect_changed_reviews('(select coalesce(max(source_run_id), 0) from ' ~ this ~ ')') }}
        )
    {% endif %}
)

select
    l.review_id,
    s.aspect_key,
    s.prob,
    s.model_version,
    s.scored_run_id,
    {{ aspect_source_run_id() }} as source_run_id
from latest as l
left join {{ ref('stg_review_aspect_score') }} as s
    on s.review_id = l.review_id and s.timestamp_updated = l.updated_at and s.label = 1
{% if not is_incremental() %}
    where s.aspect_key is not null
{% endif %}
