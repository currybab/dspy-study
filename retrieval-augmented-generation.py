import dspy
lm = dspy.LM('openai/gpt-4o-mini')
dspy.configure(lm=lm)

qa = dspy.Predict('question: str -> response: str')
response = qa(question="what are high memory and low memory on linux?")

print(response.response)

print(dspy.inspect_history(n=1))