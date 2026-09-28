-- Descriptive before/after aggregates around each treatment candidate: the 14 days before the
-- patch against the day of the patch and the 13 days after. Not a causal estimate (Phase 4).
with patches as (
    select patch_id, game_key, cast(published_at as date) as patch_date, title, patch_type
    from {{ ref('fact_patch') }}
    where is_treatment_candidate = 1
),

windows as (
    select
        p.patch_id,
        coalesce(sum(case when s.[date] < p.patch_date then s.n_reviews end), 0) as n_pre,
        coalesce(sum(case when s.[date] < p.patch_date then s.n_positive end), 0) as n_positive_pre,
        coalesce(sum(case when s.[date] >= p.patch_date then s.n_reviews end), 0) as n_post,
        coalesce(sum(case when s.[date] >= p.patch_date then s.n_positive end), 0) as n_positive_post
    from patches as p
    left join {{ ref('daily_game_sentiment') }} as s
        on
            s.game_key = p.game_key
            and s.[date] between dateadd(day, -14, p.patch_date) and dateadd(day, 13, p.patch_date)
    group by p.patch_id
)

select
    p.patch_id,
    p.game_key,
    p.patch_date,
    p.title,
    p.patch_type,
    w.n_pre,
    w.n_post,
    case when w.n_pre > 0 then cast(w.n_positive_pre as float) / w.n_pre end as pos_share_pre,
    case when w.n_post > 0 then cast(w.n_positive_post as float) / w.n_post end as pos_share_post,
    case
        when w.n_pre > 0 and w.n_post > 0
            then cast(w.n_positive_post as float) / w.n_post
            - cast(w.n_positive_pre as float) / w.n_pre
    end as pos_share_delta
from patches as p
inner join windows as w on w.patch_id = p.patch_id
