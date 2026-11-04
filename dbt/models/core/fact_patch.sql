-- Official Steam posts with their patch-classifier label; press articles are excluded.
-- A table, rebuilt nightly: news volume is tiny and raw.steam_news is never purged.
-- A post before its game's backfill start stays listed but is never a treatment candidate: there
-- are no reviews before it to compare with (the patch-history fix, 4 Nov 2026).
select
    n.gid as patch_id,
    n.appid as game_key,
    n.gid,
    n.published_at,
    cast(convert(char(8), n.published_at, 112) as int) as date_key,
    n.title,
    n.url,
    n.patch_type,
    cast(
        case
            when n.is_treatment_candidate = 1
                and cast(n.published_at as date) >= g.backfill_start then 1
            else 0
        end as bit
    ) as is_treatment_candidate,
    n.ambiguous,
    n.classifier_version
from {{ ref('stg_steam_news') }} as n
left join {{ ref('dim_game') }} as g
    on g.game_key = n.appid
where n.classification_rule <> 'not_official'
