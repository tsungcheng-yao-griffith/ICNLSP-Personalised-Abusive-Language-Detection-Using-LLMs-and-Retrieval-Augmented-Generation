import openai
import pandas as pd
import csv

client = openai.OpenAI(api_key="your key here")


def decion_tree(input):
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        temperature=0,
        messages=[
            {"role": "system", "content":
                """
           You are a classifying bot that helps users classify sentences. Please analyze the following sentence and determine if it is abusive or not. A sentence is considered abusive if a reader may feel uncomfortable after reading it. Provide a score from 0 to 1, where 0 means absolutely not harmful, and 1 means definitely harmful. Lastly, just return the score and do not provide any explanation.

                Additional Knowledge Base:
           In addition, the decision could be made based on the textual message and users' attributes. Here are the results based on the decision tree mined from user attributes and their impact on identifying harmful messages. Class 0 refers to Not Harmful, and Class 1 refers to Harmful.

           Rule 1
           Conditions:
           Flag_Low_SD: is not presented
           Flag_Average_Lower_SD: is not presented
           Flag_Average_Higher_Rationality: is not presented
           Flag_Very_High_Rationality: is not presented
           Flag_Average_Lower_Irrationality: is presented
           Flag_Very_Low_Rationality: is presented
           Flag_Ex_High_SD: is presented
           Flag_Average_Lower_Rationality: is presented
           Class Distribution: [213, 273]
           Probabilities: [0.4383, 0.5617]
           Conclusion: Class 1 (harmful)
           Explanation: Users with very low rationality, high standard deviation, and average lower irrationality tend to have harmful messages.

            Rule 2
            Conditions:

            Flag_Low_SD: is not presented
            Flag_Average_Lower_SD: is not presented
            Flag_Average_Higher_Rationality: is not presented
            Flag_Very_High_Rationality: is not presented
            Flag_Average_Lower_Irrationality: is not presented
            Flag_Very_High_SD: is not presented
            Flag_Low_Rationality: is presented
            Flag_Ex_High_SD: is presented
            Class
            Distribution: [139, 11]
            Probabilities: [0.9267, 0.0733]

            Conclusion: Class 0(not harmful)
            Explanation: Users with low rationality and extremely high standard deviation are mostly associated with not harmful messages.

            Rule 3
            Conditions:

            Flag_Low_SD: is not presented
            Flag_Average_Lower_SD: is not presented
            Flag_Average_Higher_Rationality: is not presented
            Flag_Very_High_Rationality: is not presented
            Flag_Average_Lower_Irrationality: is presented
            Flag_Very_High_SD: is not presented
            Flag_Low_Rationality: is not presented
            Flag_High_Irrationality: is presented
            Flag_Average_Lower_Rationality: is not presented
            Class
            Distribution: [147, 20]
            Probabilities: [0.8802, 0.1198]

            Conclusion: Class 0(not harmful)

            Explanation: Users with high irrationality and average lower irrationality are mostly not harmful.

            Rule 4 Conditions:

            Flag_Low_SD: is not presented
            Flag_Average_Lower_SD: is not presented
            Flag_Average_Higher_Rationality: is not presented
            Flag_Very_High_Rationality: is not presented
            Flag_Very_High_SD: is presented
            Flag_Low_Rationality: is not presented
            Flag_High_Irrationality: is not presented
            Class
            Distribution: [154, 6]
            Probabilities: [0.9625, 0.0375]

            Conclusion: Class 0(not harmful)

            Explanation: Users  with very high standard deviation and no specific irrationality flags are generally not harmful.

            Rule 5
            Conditions:

            Flag_Low_SD: is not presented
            Flag_Average_Lower_SD: is not presented
            Flag_Average_Higher_Rationality: is not presented
            Flag_Very_High_Rationality: is not presented
            Flag_Very_High_SD: is presented
            Flag_Low_Rationality: is presented
            Class
            Distribution: [32, 126]
            Probabilities: [0.2025, 0.7975]

            Conclusion: Class 1(harmful)

            Explanation: Users
            with very high standard deviation and low rationality are mostly associated with harmful messages.

            Rule 6
            Conditions:

            Flag_Low_SD: is not presented
            Flag_Average_Lower_SD: is not presented
            Flag_Average_Higher_Rationality: is not presented
            Flag_Very_High_Rationality: is presented
            Flag_Very_High_SD: is not presented
            Flag_Low_Rationality: is not presented
            Flag_Very_High_Irrationality: is not presented
            Class
            Distribution: [68, 95]
            Probabilities: [0.4172, 0.5828]

            Conclusion: Class 1(harmful)

            Explanation: Users with very high rationality and no other specific flags tend to be slightly more associated with harmful messages.

            Rule 7
            Conditions:

            Flag_Low_SD: is not presented
            Flag_Average_Lower_SD: is presented
            Flag_Very_Low_Rationality: is not presented
            Flag_High_Rationality: is not presented
            Flag_Very_High_Rationality: is not presented
            Flag_Average_Higher_Rationality: is not presented
            Flag_Average_Higher_Irrationality: is not presented
            Flag_Low_Rationality: is presented
            Class
            Distribution: [416, 68]
            Probabilities: [0.8595, 0.1405]

            Conclusion: Class 0(not harmful)

            Explanation: Users with average lower standard deviation and low rationality are mostly associated with not harmful messages.

            Rule 8
            Conditions:

            Flag_Low_SD: is not presented
            Flag_Average_Lower_SD: is presented
            Flag_Very_Low_Rationality: is not presented
            Flag_High_Rationality: is not presented
            Flag_Very_High_Rationality: is not presented
            Flag_Average_Higher_Rationality: is not presented
            Flag_Average_Lower_Irrationality: is presented
            Class
            Distribution: [1092, 179]
            Probabilities: [0.8592, 0.1408]

            Conclusion: Class 0(not harmful)

            Explanation: Users with average lower standard deviation and average lower irrationality are mostly not harmful.

            Rule 9
            Conditions:

            Flag_Low_SD: is not presented
            Flag_Average_Lower_SD: is presented
            Flag_Very_Low_Rationality: is not presented
            Flag_High_Rationality: is not presented
            Flag_Very_High_Rationality: is not presented
            Class
            Distribution: [164, 2]
            Probabilities: [0.9880, 0.0120]

            Conclusion: Class 0(not harmful)

            Explanation: Users with average lower standard deviation and no other specific flags are mostly associated with not harmful messages.

            Rule 10
            Conditions:

            Flag_Low_SD: is not presented
            Flag_Average_Lower_SD: is presented
            Flag_Very_Low_Rationality: is not presented
            Flag_High_Rationality: is not presented
            Flag_Very_High_Rationality: is presented
            Class
            Distribution: [520, 114]
            Probabilities: [0.8202, 0.1798]

            Conclusion: Class 0(not harmful)

            Explanation: Users with average lower standard deviation and very high rationality are generally not harmful.

            Rule 11
            Conditions:

            Flag_Low_SD: is not presented
            Flag_Average_Lower_SD: is presented
            Flag_Very_Low_Rationality: is not presented
            Flag_High_Rationality: is presented
            Flag_Average_Higher_Irrationality: is not presented
            Class
            Distribution: [710, 75]
            Probabilities: [0.9045, 0.0955]

            Conclusion: Class 0(not harmful)

            Explanation: Users with average lower standard deviation and high rationality are mostly associated with not harmful messages.

            Rule 12
            Conditions:

            Flag_Low_SD: is presented
            Flag_High_Rationality: is not presented
            Flag_High_SD: is not presented
            Flag_Low_Irrationality: is presented
            Class
            Distribution: [23, 143]
            Probabilities: [0.1389, 0.8611]

            Conclusion: Class 1(harmful)

            Explanation: Users with low standard deviation and low irrationality are mostly associated with harmful messages.

            Rule 13
            Conditions:

            Flag_Low_SD: is presented
            Flag_High_Rationality: is presented
            Flag_High_SD: is not presented
            Flag_High_Irrationality: is presented
            Class
            Distribution: [428, 60]
            Probabilities: [0.8776, 0.1224]

            Conclusion: Class 0(not harmful)

            Explanation: Users with low standard deviation, high rationality, and high irrationality are generally not harmful.

                """
             },
            {"role": "user", "content": str(input)}
        ]
    )
    return completion.choices[0].message.content


def associating_rule_mining(input):
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        temperature=0,
        messages=[
            {"role": "system", "content":
                """
           You are a classifying bot that helps users classify sentences. Please analyze the following sentence and determine if it is abusive or not. A sentence is considered abusive if a reader may feel uncomfortable after reading it. Provide a score from 0 to 1, where 0 means absolutely not harmful, and 1 means definitely harmful. Lastly, just return the score and do not provide any explanation.

               Additional Knowledge Base:
               In addition, the decision could be made based on the textual message and users' attributes. Here are the association rules mined from user attributes and their impact on identifying harmful messages:
Rule 1:
Antecedents: Flag_Low_Rationality, Flag_Ex_High_SD
Consequents: discomfort
Support: 0.0014
Confidence: 0.78
Lift: 2.31
Explanation: When users exhibit low rationality and extremely high standard deviation in identity, there is a strong association with message content leading to discomfort (discomfort), with a lift of 2.31.

Rule 2:
Antecedents: Flag_Very_High_Irrationality, Flag_Low_Rationality
Consequents: discomfort
Support: 0.0025
Confidence: 0.72
Lift: 2.13
Explanation: The combination of very high irrationality and low rationality significantly correlates with messages causing discomfort (discomfort), with a lift of 2.13.

Rule 3:
Antecedents: Flag_Very_High_Irrationality, Flag_Low_Rationality, Flag_Ex_High_SD
Consequents: discomfort
Support: 0.0014
Confidence: 0.78
Lift: 2.31
Explanation: When users display very high irrationality, low rationality, and extremely high standard deviation in identity, there is a strong association with message content leading to discomfort (discomfort), with a lift of 2.31.

Rule 4:
Antecedents: Flag_Low_Rationality, Flag_Ex_High_SD
Consequents: discomfort
Support: 0.0014
Confidence: 0.78
Lift: 19.00
Explanation: Low rationality and extremely high standard deviation in identity are strongly associated with very high irrationality and messages causing discomfort (discomfort), with a lift of 19.00.

Rule 5:
Antecedents: Flag_Very_High_Irrationality, Flag_Very_High_SD, Flag_Low_Rationality
Consequents: discomfort
Support: 0.0012
Confidence: 0.66
Lift: 1.95
Explanation: The combination of very high irrationality, very high standard deviation in identity, and low rationality moderately correlates with messages causing discomfort (discomfort), with a lift of 1.95.

Rule 6:
Antecedents: Flag_Average_Lower_SD
Consequents: comfort
Support: 0.090952143
Confidence: 0.737309825
Lift: 1.114630403
Explanation: The combination of average lower standard deviation(Flag_Average_Lower_SD) significantly correlates with comfort, with a lift of 1.114630403.

Rule 7:
Antecedents: Flag_Low_SD
Consequents: comfort
Support: 0.078077999
Confidence: 0.920206718
Lift: 1.391125346

Explanation: The combination of low standard deviation(Flag_Low_SD) significantly correlates with comfort, with a lift of 1.391125346.

Rule 8:
Antecedents: Flag_Average_Lower_Irrationality
Consequents: comfort
Support: 0.098529296
Confidence: 0.708564581
Lift: 1.071174692

Explanation: The combination of average lower irrationality(Flag_Average_Lower_Irrationality) significantly correlates with comfort, with a lift of 1.071174692.

Rule 9:
Antecedents: Flag_High_Rationality
Consequents: comfort
Support: 0.086067335
Confidence: 0.700799771
Lift: 1.059436217

Explanation: The combination of high rationality(Flag_High_Rationality) significantly correlates with comfort, with a lift of 1.059436217.

                """
             },
            {"role": "user", "content": str(input)}
        ]
    )
    return completion.choices[0].message.content


def no_knowledge(input):
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        temperature=0,
        messages=[
            {"role": "system", "content":
                """
           You are a classifying bot that helps users classify sentences. Please analyze the following sentence and determine if it is abusive or not. A sentence is considered abusive if a reader may feel uncomfortable after reading it. Provide a score from 0 to 1, where 0 means absolutely not harmful, and 1 means definitely harmful. Lastly, just return the score and do not provide any explanation.

                """
             },
            {"role": "user", "content": str(input)}
        ]
    )
    return completion.choices[0].message.content





