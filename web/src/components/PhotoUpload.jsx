export default function PhotoUpload({
  room,
  hasPhoto,
  onPhoto,
  onClear,
  busy,
  corroboration,
}) {
  return (
    <div className="photobar">
      <label className="filebtn">
        {hasPhoto ? 'Replace wall photo' : 'Add wall photo'}
        <input
          type="file"
          accept="image/*"
          disabled={busy}
          onChange={(event) => {
            const file = event.target.files?.[0]
            if (file) onPhoto(room, file)
            event.target.value = ''
          }}
        />
      </label>
      {hasPhoto && (
        <button onClick={() => onClear(room)} disabled={busy}>
          Remove
        </button>
      )}
      <span className="meta" style={{ margin: 0 }}>
        {corroboration && corroboration !== 'no photo'
          ? corroboration
          : 'A photo can confirm an alert, never cancel one.'}
      </span>
    </div>
  )
}
