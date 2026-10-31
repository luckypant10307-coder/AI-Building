import { useCallback, useEffect, useState } from 'react'
import ComponentStatus from './components/ComponentStatus'
import RoomCard from './components/RoomCard'
import { clearPhoto, getAlerts, getComponents, runDemo, uploadPhoto } from './api'

const FAULTS = [
  ['', 'no fault (healthy flat)'],
  ['hidden_pipe_leak', 'hidden pipe leak'],
  ['roof_seepage', 'roof seepage'],
  ['condensation', 'condensation'],
  ['running_fixture', 'running fixture'],
]
const ROOMS = ['living', 'kitchen', 'bathroom', 'bedroom']

export default function App() {
  const [fault, setFault] = useState('hidden_pipe_leak')
  const [room, setRoom] = useState('bathroom')
  const [language, setLanguage] = useState('en')
  const [rooms, setRooms] = useState([])
  const [components, setComponents] = useState([])
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)

  useEffect(() => {
    getComponents().then(setComponents).catch(() => {})
  }, [])

  const refresh = useCallback(
    async (lang = language) => {
      const data = await getAlerts(lang)
      setRooms(data.rooms)
    },
    [language]
  )

  const run = useCallback(async () => {
    setBusy(true)
    setError(null)
    try {
      await runDemo({ fault: fault || null, room })
      await refresh()
    } catch (e) {
      setError(e.message)
    } finally {
      setBusy(false)
    }
  }, [fault, room, refresh])

  useEffect(() => {
    run()
    // run once on mount; later runs are driven by the button
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const onPhoto = async (forRoom, file) => {
    setBusy(true)
    setError(null)
    try {
      await uploadPhoto(forRoom, file)
      await refresh()
    } catch (e) {
      setError(e.message)
    } finally {
      setBusy(false)
    }
  }

  const onClearPhoto = async (forRoom) => {
    setBusy(true)
    try {
      await clearPhoto(forRoom)
      await refresh()
    } catch (e) {
      setError(e.message)
    } finally {
      setBusy(false)
    }
  }

  const onLanguage = async (next) => {
    setLanguage(next)
    setBusy(true)
    try {
      await refresh(next)
    } catch (e) {
      setError(e.message)
    } finally {
      setBusy(false)
    }
  }

  const raised = rooms.filter((r) => r.alert.level !== 'none').length

  return (
    <div className="wrap">
      <h1>Smart Property AI</h1>
      <div className="banner">
        Sensor readings are simulated. The damp classifier and crack
        segmentation run on models trained on real photographs; everything the
        sensors report comes from the simulator, not a real flat.
      </div>

      <div className="controls">
        <select value={fault} onChange={(e) => setFault(e.target.value)}>
          {FAULTS.map(([value, label]) => (
            <option key={value} value={value}>
              {label}
            </option>
          ))}
        </select>
        <select
          value={room}
          onChange={(e) => setRoom(e.target.value)}
          disabled={!fault}
        >
          {ROOMS.map((r) => (
            <option key={r} value={r}>
              {r}
            </option>
          ))}
        </select>
        <select value={language} onChange={(e) => onLanguage(e.target.value)}>
          <option value="en">English</option>
          <option value="hi">हिन्दी</option>
        </select>
        <button className="primary" onClick={run} disabled={busy}>
          {busy ? 'Working…' : 'Run'}
        </button>
      </div>

      <ComponentStatus components={components} />

      {error && <div className="err">{error}</div>}

      <h2>
        {rooms.length
          ? `${raised} of ${rooms.length} rooms raised`
          : 'No readings loaded'}
      </h2>
      <div className="grid">
        {rooms.map((result) => (
          <RoomCard
            key={result.room}
            result={result}
            onPhoto={onPhoto}
            onClearPhoto={onClearPhoto}
            busy={busy}
          />
        ))}
      </div>

      <div className="foot">
        Add a wall photo to any room to bring components 5 and 7 into the
        decision. A photo corroborates a sensor alert; it never cancels one,
        because the sensors read moisture inside the plaster days before
        anything shows on the surface.
      </div>
    </div>
  )
}
