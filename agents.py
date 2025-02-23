import openai

# Set OpenAI API Key
OPENAI_API_KEY = "Your Api Key"
client = openai.OpenAI(api_key=OPENAI_API_KEY)


# Convert anomalies into questions for OpenAI processing
def formulate_questions(anomalies, ticker_symbol):
    question = (
            f"On {anomalies}, for the {ticker_symbol} stock price there was an unusual fluctuation. "
            f"Find the current affair news around {anomalies}. Find 3 questions to ask the agents given the current affairs of {ticker_symbol} on {anomalies} ."
        )
        
    return question

# Web Research Agent using OpenAI API
def web_research_agent(question):
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": question}
        ]
    )
    return completion.choices[0].message.content

# Institutional Knowledge Agent using OpenAI API
def institutional_knowledge_agent(question):
  prompt = f"""You are a  stock market expert to verify some anomaly data-related questions. 
        What do you think given the posed questions below is the specific reason for the following stocks price fluctuation? Find me specific reasons for the following anomalies.
        Question: {question} .Limit the answer to 200 words and point wise concise but particular answers. Dont display the questions in the answer. just use the questions for guidance.
        """
  completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a financial market expert."},
            {"role": "user", "content": prompt}
        ]
    )
  return completion.choices[0].message.content

# Consolidate responses from all agents into a single report
def consolidate_reports(web_response, institutional_response):
  prompt = f"""
    Your role is to summarise reports from several experts to present to management.
    Each expert tried to answer data-related questions provided below.
    Please concisely summarise the experts' answers.
    What do you think given the posed questions below is the specific reason for the following stocks price fluctuation? Find me specific reasons for the following anomalies.

    Web Research Results: {web_response}
    Institutional Knowledge Results: {institutional_response}

    Limit the answer to 200 words and point wise concise but particular answers. Dont display the questions in the answer. just use the questions for guidance.
    """
  response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a report consolidation expert."},
            {"role": "user", "content": prompt}
        ]
    )
  return response.choices[0].message.content

# Simulate management discussion using another OpenAI API call
def management_discussion(consolidated_report):
    prompt = f"""
    As a panel of management agents, please review and discuss the following consolidated report on S&P 500 Index anomalies. 
    What do you think given the posed questions below is the specific reason for the following stocks price fluctuation? Find me specific reasons for the following anomalies.

    Consolidated Report: {consolidated_report}

Limit the answer to 200 words and point wise concise but particular answers and give specific reasons. Dont display the questions in the answer. just use the questions for guidance.

    """

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a panel of management experts discussing a financial report."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

def buy_sell_decision(web_response):
    prompt = f"""
    As a panel of management agents, please review and discuss the reasons behind the anomalies. 
    After gathering insights from financial market, macroeconomic, and statistical perspectives. based on the understanding you would Buy or Sell?
    First answer only one word: Buy or Sell and then very concise reasoning in three lines.
    Web response report: {web_response}
    """

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a panel of management experts discussing a financial report."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content
