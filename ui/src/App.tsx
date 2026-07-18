const foundationItems = [
  "Typed evidence and session contracts",
  "Deterministic safety state machine",
  "Reproducible experiment manifests",
];

export default function App() {
  return (
    <main className="shell">
      <p className="eyebrow">Research interface foundation</p>
      <h1>NeuroSelect</h1>
      <p className="summary">
        Personalized, uncertainty-aware language prediction for reducing BCI
        selections.
      </p>

      <section aria-labelledby="foundation-heading" className="panel">
        <h2 id="foundation-heading">Foundation ready</h2>
        <ul>
          {foundationItems.map((item) => (
            <li key={item}>{item}</li>
          ))}
        </ul>
      </section>

      <aside className="boundary" aria-label="Research boundary">
        NeuroSelect does not decode unrestricted thoughts. This shell is not yet
        an operational BCI interface.
      </aside>
    </main>
  );
}
