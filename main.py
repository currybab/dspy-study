import dspy
import os

lm = dspy.LM("openai/gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
dspy.configure(lm=lm)


from typing import Literal

# class Emotion(dspy.Signature):
#     """Classify sentiment of a given sentence."""

#     sentence: str = dspy.InputField()
#     sentiment: Literal['sadness', 'joy', 'love', 'anger', 'fear', 'surprise'] = dspy.OutputField()

# sentence = "i started feeling a little vulnerable when the giant spotlight started blinding me"  # dair-ai/emotion에서

# classify = dspy.Predict(Emotion)
# print(classify(sentence=sentence))

# class CheckCitationFaithfulness(dspy.Signature):
#     """텍스트가 제공된 컨텍스트에 기반하는지 확인합니다."""

#     context: str = dspy.InputField(desc="여기의 사실들은 참이라고 가정됩니다")
#     text: str = dspy.InputField()
#     faithfulness: bool = dspy.OutputField()
#     evidence: dict[str, list[str]] = dspy.OutputField(desc="주장을 뒷받침하는 증거")

# context = "The 21-year-old made seven appearances for the Hammers and netted his only goal for them in a Europa League qualification round match against Andorran side FC Lustrains last season. Lee had two loan spells in League One last term, with Blackpool and then Colchester United. He scored twice for the U's but was unable to save them from relegation. The length of Lee's contract with the promoted Tykes has not been revealed. Find all the latest football transfers on our dedicated page."

# text = "Lee scored 3 goals for Colchester United."

# faithfulness = dspy.ChainOfThought(CheckCitationFaithfulness)
# print(faithfulness(context=context, text=text))

# class DogPictureSignature(dspy.Signature):
#     """Output the dog breed of the dog in the image."""
#     image_1: dspy.Image = dspy.InputField(desc="An image of a dog")
#     answer: str = dspy.OutputField(desc="The dog breed of the dog in the image")

# image_url = "https://picsum.photos/id/237/200/300"
# classify = dspy.Predict(DogPictureSignature)
# print(classify(image_1=dspy.Image.from_url(image_url)))

# question = "ColBERT 검색 모델의 훌륭한 점은 무엇인가요?"

# # 1) 시그니처로 선언하고 구성을 전달합니다.
# classify = dspy.ChainOfThought('question -> answer', n=5)

# # 2) 입력 인수로 호출합니다.
# response = classify(question=question)

# # 3) 출력에 접근합니다.
# print(response.completions.answer)
# print(f"추론: {response.reasoning}")
# print(f"답변: {response.answer}")

article_summary = dspy.Example(article="이것은 기사입니다.", summary="이것은 요약입니다.").with_inputs("article")

input_key_only = article_summary.inputs()
non_input_key_only = article_summary.labels()

print("입력 필드만 있는 Example 객체:", input_key_only)
print("비입력 필드만 있는 Example 객체:", non_input_key_only)