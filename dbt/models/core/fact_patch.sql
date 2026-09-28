-- Official Steam posts with their patch-classifier label; press articles are excluded.
-- A table, rebuilt nightly: news volume is tiny and raw.steam_news is never purged.
select
    n.gid as patch_id,
    n.appid as game_key,
    n.gid,
    n.published_at,
    cast(convert(char(8), n.published_at, 112) as int) as date_key,
    n.title,
    n.url,
    n.patch_type,
    n.is_treatment_candidate,
    n.ambiguous,
    n.classifier_version
from {{ ref('stg_steam_news') }} as n
where n.classification_rule <> 'not_official'
