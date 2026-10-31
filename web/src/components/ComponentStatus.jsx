export default function ComponentStatus({ components }) {
  if (!components?.length) return null
  const live = components.filter((c) => c.live).length
  return (
    <>
      <h2>
        Components live: {live} of {components.length}
      </h2>
      <div className="chips">
        {components.map((c) => (
          <span
            key={`${c.id}-${c.name}`}
            className={`chip ${c.live ? 'live' : 'off'}`}
            title={c.note || c.how}
          >
            {c.id !== '-' ? `${c.id}. ` : ''}
            {c.name}
          </span>
        ))}
      </div>
      {components.some((c) => !c.live) && (
        <div className="meta">
          Not running:{' '}
          {components
            .filter((c) => !c.live)
            .map((c) => `${c.name} (${c.note || 'unavailable'})`)
            .join('; ')}
        </div>
      )}
    </>
  )
}
