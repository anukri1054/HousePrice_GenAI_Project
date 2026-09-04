import google.generativeai as genai


def initialize_gemini(api_key):
    genai.configure(api_key=api_key)


def generate_house_analysis(
    house_features,
    predicted_price,
    feature_importance
):

    model = genai.GenerativeModel("gemini-1.5-flash")

    prompt = f"""
You are an expert AI Real Estate Analyst.

A machine learning model has predicted the price of a house.

Analyze the following property information:

Median Income: {house_features['MedInc']}

House Age: {house_features['HouseAge']}

Average Rooms: {house_features['AveRooms']}

Average Bedrooms: {house_features['AveBedrms']}

Population: {house_features['Population']}

Average Occupancy: {house_features['AveOccup']}

Latitude: {house_features['Latitude']}

Longitude: {house_features['Longitude']}


Predicted House Price:

${predicted_price:,.2f}


Important Predictive Factors:

{feature_importance}


Generate a structured professional report with the following sections:

1. Executive Summary
2. Price Prediction Explanation
3. Important Factors Affecting Price
4. Positive Factors
5. Limitations and Risks
6. Final Conclusion

Rules:

- Do not guarantee the predicted price.
- Clearly mention that this is a machine learning estimate.
- Use professional but simple language.
- Do not invent information that was not provided.
"""

    response = model.generate_content(prompt)

    return response.text