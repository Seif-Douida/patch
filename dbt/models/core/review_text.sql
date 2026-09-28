{{ config(
    materialized='incremental',
    unique_key='review_id',
    incremental_strategy='merge',
    full_refresh=false
) }}

-- Review text, kept apart so fact_review stays narrow (spec §6.4).
with changed as (
    select distinct recommendation_id
    from {{ source('raw', 'steam_review') }}
    {% if is_incremental() %}
        where last_seen_run_id > (select coalesce(max(last_seen_run_id), 0) from {{ this }})
    {% endif %}
)

select
    r.review_id,
    r.review_text as [text],
    r.last_seen_run_id
from {{ ref('stg_steam_review') }} as r
inner join changed as c on c.recommendation_id = r.review_id
