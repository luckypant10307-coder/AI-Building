import PhotoUpload from './PhotoUpload'

const days = (value) =>
  value === null || value === undefined ? null : `${value.toFixed(1)} d`

function Forecast({ forecast }) {
  if (!forecast) return null
  const damp = forecast.visible_damp
  const mould = forecast.mould_risk
  return (
    <div className="row">
      <div className="stat">
        <b>{days(damp?.days) ?? 'none'}</b>
        visible damp {damp?.days == null && '(not within 7 d)'}
      </div>
      <div className="stat">
        <b>{days(mould?.days) ?? 'none'}</b>
        mould risk {mould?.days == null && '(not within 7 d)'}
      </div>
      <div className="stat">
        <b>{forecast.backend}</b>
        forecaster
      </div>
    </div>
  )
}

function PhotoEvidence({ photo }) {
  if (!photo) return null
  const bits = []
  if (photo.top_class) {
    bits.push(
      `classifier: ${photo.top_class}` +
        (photo.damp_probability != null
          ? ` (damp ${(photo.damp_probability * 100).toFixed(0)}%)`
          : '')
    )
  }
  if (photo.crack_detected !== undefined) {
    bits.push(
      photo.crack_detected
        ? `crack found (${(photo.crack_confidence * 100).toFixed(0)}% confident)`
        : 'no crack found'
    )
  }
  if (photo.crack_mm != null) {
    bits.push(
      `widest ${photo.crack_mm.toFixed(2)} mm` +
        (photo.crack_width_trustworthy ? '' : ' (stand-in mask, not reliable)')
    )
  }
  if (photo.crack_growth_mm != null) {
    bits.push(`grew ${photo.crack_growth_mm.toFixed(2)} mm since last visit`)
  }
  return bits.length ? <div className="meta">Photo — {bits.join(' · ')}</div> : null
}

export default function RoomCard({ result, onPhoto, onClearPhoto, busy }) {
  const { room, alert, forecast, advice, photo } = result
  const raised = alert.level !== 'none'

  return (
    <div className="card">
      <div className="head">
        <span className="room">{room}</span>
        <span className={`level ${alert.level}`}>{alert.level}</span>
      </div>

      {raised ? (
        <>
          <div className="meta">
            {alert.kind?.replace(/_/g, ' ')} · confidence{' '}
            {Math.round(alert.confidence * 100)}% · anomalous hours{' '}
            {Math.round(alert.anomaly_fraction * 100)}%
          </div>
          <Forecast forecast={forecast} />
          <ul className="evidence">
            {alert.evidence_for?.map((line) => (
              <li key={line}>{line}</li>
            ))}
          </ul>
        </>
      ) : (
        <div className="meta">Nothing outside this room&rsquo;s normal week.</div>
      )}

      <PhotoEvidence photo={photo} />

      {advice && (
        <div className="advice">
          <div>
            <b>Cause.</b> {advice.cause}
          </div>
          <div>
            <b>Do.</b> {advice.action}
          </div>
          <div>
            <b>Call.</b> {advice.who_to_call}
          </div>
          <div className="meta">
            {advice.urgency} · wording from{' '}
            {advice.source === 'llm' ? 'Gemini' : 'template'}
          </div>
        </div>
      )}

      {alert.needs_technician_confirmation && (
        <div className="flag">Needs technician confirmation</div>
      )}

      <PhotoUpload
        room={room}
        hasPhoto={Boolean(photo)}
        onPhoto={onPhoto}
        onClear={onClearPhoto}
        busy={busy}
        corroboration={alert.photo_corroboration}
      />
    </div>
  )
}
