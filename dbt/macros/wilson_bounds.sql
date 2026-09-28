{#-
    Wilson score interval for a share of positive reviews (spec §6.4: 90% bounds).

        p      = positive / n
        centre = (p + z²/2n) / (1 + z²/n)
        half   = z * sqrt(p(1 - p)/n + z²/4n²) / (1 + z²/n)

    z = 1.6449 is the two-sided 90% normal quantile. n = 0 gives NULL, not a division error.
    Bounds are clamped to [0, 1]: at p = 1 floating point gives 1.0000000000000002. The clamp
    sits inside the n = 0 check because GREATEST / LEAST skip NULLs (GREATEST(0, NULL) = 0).
    Checked against statsmodels in tests/dbt/test_wilson_reference.py.
-#}
{% macro wilson_lower(positive, n, z=1.6449) -%}
    {{ _wilson(positive, n, z, '-', 'greatest', '0.0') }}
{%- endmacro %}

{% macro wilson_upper(positive, n, z=1.6449) -%}
    {{ _wilson(positive, n, z, '+', 'least', '1.0') }}
{%- endmacro %}

{% macro _wilson(positive, n, z, sign, clamp, limit) -%}
    case when coalesce({{ n }}, 0) = 0 then null else {{ clamp }}({{ limit }}, (
        (cast({{ positive }} as float) / {{ n }} + {{ z }} * {{ z }} / (2.0 * {{ n }}))
        {{ sign }} {{ z }} * sqrt(
            (cast({{ positive }} as float) / {{ n }}) * (1 - cast({{ positive }} as float) / {{ n }})
            / {{ n }}
            + {{ z }} * {{ z }} / (4.0 * {{ n }} * {{ n }})
        )
    ) / (1 + {{ z }} * {{ z }} / {{ n }})) end
{%- endmacro %}
