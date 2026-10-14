-- For each review version, its latest score, one row per aspect. Every model's scores stay in raw;
-- the latest is the one with the highest run id (the scoring stage's own rule), so a new champion's
-- score replaces the old one as soon as the stage reaches the review, and a return to an earlier
-- champion re-stamps that model's rows with the returning run.
with ranked as (
    select
        s.*,
        row_number() over (
            partition by s.recommendation_id, s.timestamp_updated
            order by s.scored_run_id desc, s.scored_at desc, s.model_version desc
        ) as score_rank
    from {{ source('raw', 'review_aspect_score') }} as s
)

select
    r.recommendation_id as review_id,
    r.timestamp_updated,
    r.model_version,
    r.scored_run_id,
    cast(a.aspect_key as nvarchar(32)) as aspect_key,
    a.prob,
    a.label
from ranked as r
cross apply (
    values
    {%- for code in var('aspects') %}
        ('{{ code }}', r.p_{{ code }}, r.l_{{ code }}){{ "," if not loop.last }}
    {%- endfor %}
) as a (aspect_key, prob, label)
where r.score_rank = 1
