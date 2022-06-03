const textArea = document.getElementById('input')
const positiveScoreBar = document.getElementById('positive-sentiment-bar-fill')
const emoji = document.getElementById('emoji')

const apiBaseUrl = 'http://localhost:4242'

const catEmoji = '🐱'
const dogEmoji = '🐶'
const questionEmoji = '❔'

const updateEmoji = (pos) => {
  if (pos > 0.7) {
      emoji.innerText = catEmoji
      result.innerText = 'cat!'
  } else if (pos < 0.3) {
      emoji.innerText = dogEmoji
      result.innerText = 'dog!'
  } else {
      emoji.innerText = questionEmoji
      result.innerText = 'giraffe?!'
  }
}

const getPrediction = async (input) => {
  const file = input.files[0]
  
  const image = document.getElementById('output')
  image.src = URL.createObjectURL(file)
  
  const formData = new FormData()
  formData.append("file", file)

  const response = await fetch(`${apiBaseUrl}/api/predict-image`, { method: 'POST', body: formData })
  const body = await response.json()
  return body.result
}

const getSentiment = async (input) => {
  const { cat, dog } = await getPrediction(input)
  if (cat == null) return
  
  updateEmoji(cat)
}

const image_input = document.querySelector("#image_input")

image_input.addEventListener("change", function() {
  const reader = new FileReader()
  reader.addEventListener("load", () => {
    const uploaded_image = reader.result
    document.querySelector("#display_image").style.backgroundImage = `url(${uploaded_image})`
  })
  reader.readAsDataUrl(this.files[0])
})
