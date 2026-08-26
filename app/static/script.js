const jobText = document.getElementById("jobText");

const analyzeBtn = document.getElementById("analyzeBtn");

const buttonText = document.getElementById("buttonText");

const loader = document.getElementById("loader");

const clearBtn = document.getElementById("clearBtn");

const exampleBtn = document.getElementById("exampleBtn");

const characterCount =
    document.getElementById("characterCount");

const placeholder =
    document.getElementById("placeholder");

const resultContent =
    document.getElementById("resultContent");

const resultCard =
    document.getElementById("resultCard");

const resultIcon =
    document.getElementById("resultIcon");

const prediction =
    document.getElementById("prediction");

const resultMessage =
    document.getElementById("resultMessage");

const confidenceValue =
    document.getElementById("confidenceValue");

const confidenceFill =
    document.getElementById("confidenceFill");


/* CHARACTER COUNT */

jobText.addEventListener("input", () => {

    const length = jobText.value.length;

    characterCount.textContent =
        `${length.toLocaleString()} characters`;

});


/* EXAMPLE */

exampleBtn.addEventListener("click", () => {

    jobText.value = `
We are looking for a motivated Python Developer
to join our growing technology team.

Responsibilities:
- Develop Python applications
- Work with REST APIs
- Build database integrations
- Collaborate with other developers

Requirements:
- 2+ years of Python experience
- SQL knowledge
- Good communication skills

Benefits:
- Health insurance
- Paid vacation
- Competitive salary
- Flexible working hours
`;

    jobText.dispatchEvent(
        new Event("input")
    );

});


/* CLEAR */

clearBtn.addEventListener("click", () => {

    jobText.value = "";

    jobText.dispatchEvent(
        new Event("input")
    );

    placeholder.classList.remove("hidden");

    resultContent.classList.add("hidden");

    resultCard.classList.remove(
        "fake-result",
        "real-result"
    );

});


/* ANALYZE */

analyzeBtn.addEventListener(
    "click",
    async () => {

        const text = jobText.value.trim();

        if (!text) {

            alert(
                "Please enter a job posting first."
            );

            return;
        }


        /* LOADING STATE */

        analyzeBtn.disabled = true;

        buttonText.textContent =
            "Analyzing...";

        loader.classList.remove("hidden");


        try {

            const response =
                await fetch("/predict", {

                    method: "POST",

                    headers: {
                        "Content-Type":
                            "application/json"
                    },

                    body: JSON.stringify({
                        text: text
                    })

                });


            if (!response.ok) {

                throw new Error(
                    "Prediction request failed."
                );

            }


            const data =
                await response.json();


            showResult(data);


        } catch (error) {

            console.error(error);

            alert(
                "Unable to analyze the job posting. Please try again."
            );

        } finally {

            analyzeBtn.disabled = false;

            buttonText.textContent =
                "Analyze Job";

            loader.classList.add("hidden");

        }

    }
);


/* SHOW RESULT */

function showResult(data) {

    const isFake =
        data.prediction === "fake";


    const confidence =
        Number(data.confidence) * 100;


    placeholder.classList.add("hidden");

    resultContent.classList.remove("hidden");


    resultCard.classList.remove(
        "fake-result",
        "real-result"
    );


    resultCard.classList.add(
        isFake
            ? "fake-result"
            : "real-result"
    );


    if (isFake) {

        resultIcon.textContent = "⚠";

        prediction.textContent =
            "FAKE JOB";

        resultMessage.textContent =
            "This posting shows patterns associated with potentially fraudulent job listings.";

    } else {

        resultIcon.textContent = "✓";

        prediction.textContent =
            "REAL JOB";

        resultMessage.textContent =
            "This posting appears to be legitimate based on the model's analysis.";

    }


    confidenceValue.textContent =
        `${confidence.toFixed(1)}%`;


    confidenceFill.style.width =
        `${confidence}%`;

}