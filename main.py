from translator import english_to_marshallese, marshallese_to_english


if __name__ == "__main__":
    # Test English to Marshallese
    print("English to Marshallese:")
    english_text = "Hello, do you have a headache or pain in your body?"
    marshallese = english_to_marshallese(english_text)
    print(f"Input: {english_text}")
    print(f"Output: {marshallese}\n")

    # Test Marshallese to English
    print("Marshallese to English:")
    marshallese_text = "Iakwe, do you have a Metak bar or Metak in your body?"
    english = marshallese_to_english(marshallese_text)
    print(f"Input: {marshallese_text}")
    print(f"Output: {english}\n")
