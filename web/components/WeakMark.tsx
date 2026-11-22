/** The less reliable aspects' marker: a visible asterisk, and words for screen readers. */
export function WeakMark() {
  return (
    <>
      <span aria-hidden="true">*</span>
      <span className="visually-hidden"> (detected less reliably)</span>
    </>
  );
}
