const json = async (res) => {
  const body = await res.json().catch(() => ({}))
  if (!res.ok) throw new Error(body.error || `${res.status} ${res.statusText}`)
  return body
}

export const getComponents = () => fetch('/api/components').then(json)

export const runDemo = (body) =>
  fetch('/api/demo', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  }).then(json)

export const getAlerts = (language) =>
  fetch(`/api/alerts?language=${encodeURIComponent(language)}`).then(json)

export const uploadPhoto = (room, file) => {
  const form = new FormData()
  form.append('room', room)
  form.append('photo', file)
  return fetch('/api/photo', { method: 'POST', body: form }).then(json)
}

export const clearPhoto = (room) =>
  fetch('/api/photo', {
    method: 'DELETE',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ room }),
  }).then(json)
