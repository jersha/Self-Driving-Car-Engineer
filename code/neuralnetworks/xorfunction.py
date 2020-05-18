import pandas as pd

#Set weight1, weight2, and bias
WeightOR1 = 0.1
WeightOR2 = 0.1
BiasOR = -0.1

WeightAND1 = 0.1
WeightAND2 = 0.1
BiasAND = -0.2

WeightNOT = -0.1
BiasNOT = 0.0

#Inputs and Outputs
TestInputs = [(0, 0), (0, 1), (1, 0), (1, 1)]
CorrectOutputs = [False, True, True, False]
Outputs = []

# Generate and check output
for TestInput, CorrectOutput in zip(TestInputs, CorrectOutputs):
    LinearCombinationOR = TestInput[0] * WeightOR1 + TestInput[1] * WeightOR2 + BiasOR
    LinearCombinationAND = TestInput[0] * WeightAND1 + TestInput[1] * WeightAND2 + BiasAND
    OutputOR = int(LinearCombinationOR >= 0)
    OutputAND = int(LinearCombinationAND >= 0)
    LinearCombinationNAND = OutputAND * WeightNOT + BiasNOT
    OutputNAND = int(LinearCombinationNAND >= 0)
    LinearCombinationXOR = OutputOR * WeightAND1 + OutputNAND * WeightAND2 + BiasAND
    OutputXOR = int(LinearCombinationXOR >= 0)
    IsCorrectString = 'Yes' if OutputXOR == CorrectOutput else 'No'
    Outputs.append([TestInput[0], TestInput[1], LinearCombinationXOR, OutputXOR, IsCorrectString])
    
# Print output
NumWrong = len([Output[4] for Output in Outputs if Output[4] == 'No'])
OutputFrame = pd.DataFrame(Outputs, columns=[
        'Input 1', '  Input 2',  '  Linear Combination', '  Activation Output', '  Is Correct'])

if not NumWrong: 
    print('Nice! You got it all correct.\n')
else:
    print('You got {} wrong.  Keep trying!\n'.format(NumWrong))
print(OutputFrame.to_string(index = False))