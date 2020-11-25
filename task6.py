import task5, RelevanceFeedbackSystemNBModel
def main():
  method = int(input(" 1. Probabilistic Relevant Feedback\n 2. Classifier-based relevance feedback\nWhat method do you want to use: "))
  if method == 1:
    RelevanceFeedbackSystemNBModel.main()
  elif method == 2:
    task5.main()
if __name__=="__main__":
    main()