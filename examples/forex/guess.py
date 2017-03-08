import os
import sys
import numpy as np
from hmm.hmm import DiscreteHMM

TUNE_NUM_TRAIN_SEQ = int(sys.argv[1])

# Mapping between raw observed datas and features
EPS = 0.00001
MEAN = 1.085412019 # use raw price, instead of difference between price
STD = 0.00134864094319
SSTD = STD * 0.5
print('SSTD',SSTD)

def obs_2_id(obs):
    if obs > MEAN + 5*SSTD:
        return 10
    elif obs > MEAN + 4*SSTD:
        return 9
    elif obs > MEAN + 3*SSTD:
        return 8
    elif obs > MEAN + 2*SSTD:
        return 7
    elif obs > MEAN + 1*SSTD:
        return 6 
    elif obs < MEAN - 5*SSTD:
        return 0
    elif obs < MEAN - 4*SSTD:
        return 1
    elif obs < MEAN - 3*SSTD:
        return 2
    elif obs < MEAN - 2*SSTD:
        return 3
    elif obs < MEAN - 1*SSTD:
        return 4
    else:
        return 5

id_2_obs = {
    0: '-5',
    1: '-4',
    2: '-3',
    3: '-2',
    4: '-1',
    5: '=',
    6: '+1',
    7: '+2',
    8: '+3',
    9: '+4',
    10: '+5'
}

predict_cnt = {
    0: 0,
    1: 0,
    2: 0,
    3: 0,
    4: 0,
    5: 0,
    6: 0,
    7: 0,
    8: 0,
    9: 0,
    10: 0
}
# Reading input file
raw_seq = []
with open(os.path.join(os.path.dirname(__file__), 'input.txt')) as f:
    for line in f:
        raw_seq.append(float(line.strip()))
obs_seq = []
for r in raw_seq:
    obs_seq.append( obs_2_id(r) )

# Setting model
num_hidden_var = 6
num_obs_var = len(id_2_obs)
num_train_seq = TUNE_NUM_TRAIN_SEQ #10

# Training the model best describe the observation
good_correct = 0
good_total = 0
correct_num = 0

total = 0
correct_rise = 0
correct_down = 0
false_rise = 0
false_down = 0
for t in range(num_train_seq, len(obs_seq)-1):
    hmm = DiscreteHMM(num_hidden_var, num_obs_var)
    hmm.train(obs_seq[t-num_train_seq : t])
    p = np.array(hmm.given(obs_seq[t-num_train_seq : t])['forward'])
    belief = hmm.B.T.dot(hmm.A.T.dot(p))
    guess = np.argmax(belief)
    
    predict_cnt[guess] = predict_cnt[guess] + 1

    #np.set_printoptios(precision=2, suppress=True)
    if ( guess > obs_seq[t-1]):
        total += 1 
        if( obs_seq[t] > obs_seq[t-1]):
            correct_rise += 1
            print ('O rise, %3d -> %3d' % (obs_seq[t-1], guess))
        else:
            false_rise += 1
            print('X rise, %3d -> %3d, REAL: %3d -> %3d' % (obs_seq[t-1], guess, obs_seq[t-1], obs_seq[t]))
    elif( guess < obs_seq[t-1] ):
        total += 1 
        if( obs_seq[t] < obs_seq[t-1]):
            correct_down += 1
            print ('O down, %3d -> %3d' % (obs_seq[t-1], guess))
        else:
            false_down += 1
            print('X down, %3d -> %3d, REAL: %3d -> %3d' % (obs_seq[t-1], guess, obs_seq[t-1], obs_seq[t]))
    '''
    print('t = ', t)
    print('belief:', belief)
    print('guess : %7s' % id_2_obs[guess])
    print('target: %7s' % id_2_obs[obs_seq[t]])

    if guess == obs_seq[t]:
        print('result: correct')
        correct_num += 1
    else:
        print('result:   wrong')
    total += 1
    print('correct rate so far: %.4f' % (correct_num / total))

    if id_2_obs[guess] in ('>', '>>') and obs_seq[t] != 0:
        good_correct += int(id_2_obs[obs_seq[t]] in ('>', '>>'))
        good_total += 1
    if good_total:
        print('good guess rate so far: %.4f' % (good_correct / good_total))
    print('==================================================================')
    '''
print('==================================================================')
for keys,values in predict_cnt.items():
    print(values, end =', ')

print('correct rise: ', correct_rise)
print('correct down: ', correct_down)
print('false rise: ', false_rise)
print('false down: ', false_down)
print('Correct Rate = %.4f' % ((correct_rise + correct_down) / total))

