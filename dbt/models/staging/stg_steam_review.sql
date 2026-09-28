-- The latest version of each review, with the run that first and last saw it (any version).
with versions as (
    select
        r.*,
        row_number() over (
            partition by r.recommendation_id order by r.timestamp_updated desc
        ) as version_rank,
        min(r.first_seen_run_id) over (partition by r.recommendation_id) as review_first_seen,
        max(r.last_seen_run_id) over (partition by r.recommendation_id) as review_last_seen
    from {{ source('raw', 'steam_review') }} as r
)

select
    recommendation_id as review_id,
    appid,
    author_hash,
    author_playtime_at_review,
    language,
    review_text,
    timestamp_created,
    timestamp_updated,
    voted_up,
    votes_up,
    weighted_vote_score,
    steam_purchase,
    received_for_free,
    written_during_early_access,
    review_first_seen as first_seen_run_id,
    review_last_seen as last_seen_run_id
from versions
where version_rank = 1
