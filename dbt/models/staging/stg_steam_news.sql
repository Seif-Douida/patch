-- News items with the current classifier version's label (var `classifier_version`).
select
    n.gid,
    n.appid,
    n.title,
    n.url,
    n.published_at,
    n.feedname,
    c.patch_type,
    c.is_treatment_candidate,
    c.[rule] as classification_rule,  -- `rule` is a reserved word in T-SQL
    c.ambiguous,
    c.classifier_version
from {{ source('raw', 'steam_news') }} as n
inner join {{ source('raw', 'news_classification') }} as c
    on c.gid = n.gid and c.classifier_version = '{{ var("classifier_version") }}'
