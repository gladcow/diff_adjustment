#!/usr/bin/env python3

import argparse
import datetime
import math
import random
import statistics
import sys
import time
from collections import namedtuple
from functools import partial


def bits_to_target(bits):
    size = bits >> 24
    assert size <= 0x1d

    word = bits & 0x00ffffff
    assert 0x8000 <= word <= 0x7fffff

    if size <= 3:
        return word >> (8 * (3 - size))
    else:
        return word << (8 * (size - 3))


MAX_BITS = 0x1d00ffff
MAX_TARGET = bits_to_target(MAX_BITS)


def target_to_bits(target):
    assert target > 0
    if target > MAX_TARGET:
        print('Warning: target went above maximum ({} > {})'
              .format(target, MAX_TARGET), file=sys.stderr)
        target = MAX_TARGET
    size = (target.bit_length() + 7) // 8
    mask64 = 0xffffffffffffffff
    if size <= 3:
        compact = (target & mask64) << (8 * (3 - size))
    else:
        compact = (target >> (8 * (size - 3))) & mask64

    if compact & 0x00800000:
        compact >>= 8
        size += 1

    assert compact == (compact & 0x007fffff)
    assert size < 256
    return compact | size << 24


def bits_to_work(bits):
    return (2 << 255) // (bits_to_target(bits) + 1)


def target_to_hex(target):
    h = hex(target)[2:]
    return '0' * (64 - len(h)) + h


TARGET_1 = bits_to_target(486604799)

TARGET_BLOCK_TIME = 150

INITIAL_TIMESTAMP = 1503430225
INITIAL_HASHRATE = 500    # In PH/s.
INITIAL_DASH_TARGET = int(pow(2, 256) // (INITIAL_HASHRATE * 1e15) // TARGET_BLOCK_TIME)
INITIAL_DASH_BITS = target_to_bits(INITIAL_DASH_TARGET)
INITIAL_HEIGHT = 481824
INITIAL_SINGLE_WORK = bits_to_work(INITIAL_DASH_BITS)

State = namedtuple('State', 'height wall_time timestamp bits chainwork '
                   'hashrate msg')

states = []


def print_headers():
    print(', '.join(['Height', 'Block Time', 'Unix', 'Timestamp',
                     'Difficulty (bn)', 'Implied Difficulty (bn)',
                     'Hashrate (PH/s)', 'Comments']))


def print_state():
    state = states[-1]
    block_time = state.timestamp - states[-2].timestamp
    t = datetime.datetime.fromtimestamp(state.timestamp)
    difficulty = TARGET_1 / bits_to_target(state.bits)
    implied_diff = TARGET_1 / ((2 << 255) / (state.hashrate * 1e15 * TARGET_BLOCK_TIME))
    print(', '.join(['{:d}'.format(state.height),
                     '{:d}'.format(block_time),
                     '{:d}'.format(state.timestamp),
                     '{:%Y-%m-%d %H:%M:%S}'.format(t),
                     '{:.2f}'.format(difficulty / 1e9),
                     '{:.2f}'.format(implied_diff / 1e9),
                     '{:.0f}'.format(state.hashrate),
                     state.msg]))


def median_time_past(states):
    times = [state.timestamp for state in states]
    return sorted(times)[len(times) // 2]


def next_bits_k(msg, mtp_window, high_barrier, target_raise_frac,
                low_barrier, target_drop_frac, fast_blocks_pct):
    # Calculate N-block MTP diff
    MTP_0 = median_time_past(states[-11:])
    MTP_N = median_time_past(states[-11-mtp_window:-mtp_window])
    MTP_diff = MTP_0 - MTP_N
    bits = states[-1].bits
    target = bits_to_target(bits)

    # Long term block production time stabiliser
    t = states[-1].timestamp - states[-2017].timestamp
    if t < TARGET_BLOCK_TIME * 2016 * fast_blocks_pct // 100:
        msg.append("2016 block time difficulty raise")
        target -= target // target_drop_frac

    if MTP_diff > high_barrier:
        target += target // target_raise_frac
        msg.append("Difficulty drop {}".format(MTP_diff))
    elif MTP_diff < low_barrier:
        target -= target // target_drop_frac
        msg.append("Difficulty raise {}".format(MTP_diff))
    else:
        msg.append("Difficulty held {}".format(MTP_diff))

    return target_to_bits(target)


def suitable_block_index(index):
    assert index >= 3
    indices = [index - 2, index - 1, index]

    if states[indices[0]].timestamp > states[indices[2]].timestamp:
        indices[0], indices[2] = indices[2], indices[0]

    if states[indices[0]].timestamp > states[indices[1]].timestamp:
        indices[0], indices[1] = indices[1], indices[0]

    if states[indices[1]].timestamp > states[indices[2]].timestamp:
        indices[1], indices[2] = indices[2], indices[1]

    return indices[1]


def compute_index_fast(index_last):
    for candidate in range(index_last - 3, 0, -1):
        index_fast = suitable_block_index(candidate)
        if index_last - index_fast < 5:
            continue
        if (states[index_last].timestamp - states[index_fast].timestamp
                >= 13 * TARGET_BLOCK_TIME):
            return index_fast
    raise AssertionError('should not happen')


def compute_target(first_index, last_index):
    work = states[last_index].chainwork - states[first_index].chainwork
    work *= TARGET_BLOCK_TIME
    work //= states[last_index].timestamp - states[first_index].timestamp
    return (2 << 255) // work - 1


def next_bits_d(msg):
    N = len(states) - 1
    index_last = suitable_block_index(N)
    index_first = suitable_block_index(N - 2016)
    interval_target = compute_target(index_first, index_last)
    index_fast = compute_index_fast(index_last)
    fast_target = compute_target(index_fast, index_last)

    next_target = interval_target
    if (fast_target < interval_target - (interval_target >> 2) or
            fast_target > interval_target + (interval_target >> 2)):
        msg.append("fast target")
        next_target = fast_target
    else:
        msg.append("interval target")

    prev_target = bits_to_target(states[-1].bits)
    min_target = prev_target - (prev_target >> 3)
    if next_target < min_target:
        msg.append("min target")
        return target_to_bits(min_target)

    max_target = prev_target + (prev_target >> 3)
    if next_target > max_target:
        msg.append("max target")
        return target_to_bits(max_target)

    return target_to_bits(next_target)


def compute_cw_target(block_count):
    first, last = -1-block_count, -1
    timespan = states[last].timestamp - states[first].timestamp
    timespan = max(block_count * TARGET_BLOCK_TIME // 2, min(block_count * 2 * TARGET_BLOCK_TIME, timespan))
    work = (states[last].chainwork - states[first].chainwork) * TARGET_BLOCK_TIME // timespan
    return (2 << 255) // work - 1


def next_bits_sha(msg):
    primes = [73, 79, 83, 89, 97,
              101, 103, 107, 109, 113, 127,
              131, 137, 139, 149, 151]

    # The timestamp % len(primes) is a proxy for previous
    # block SHAx2 % len(primes), but that data is not available
    # in this simulation
    prime = primes[states[-1].timestamp % len(primes)]
    
    interval_target = compute_cw_target(prime)
    return target_to_bits(interval_target)


def next_bits_cw(msg, block_count):
    interval_target = compute_cw_target(block_count)
    return target_to_bits(interval_target)


def next_bits_wt(msg, block_count, limit_precision):
    DIFF_WEIGHT_PRECISION = 1000000

    first, last = -1-block_count, -1
    last_target = bits_to_target(states[last].bits)
    last_target_fixed = last_target // DIFF_WEIGHT_PRECISION
    timespan = 0
    prior_timestamp = states[first].timestamp
    for i in range(first + 1, last + 1):
        target_i = bits_to_target(states[i].bits)
        # Prevent negative time_i values
        timestamp = max(states[i].timestamp, prior_timestamp)
        time_i = timestamp - prior_timestamp
        prior_timestamp = timestamp
        if limit_precision:
            adj_time_i = time_i * (target_i // DIFF_WEIGHT_PRECISION) // last_target_fixed
        else:
            adj_time_i = time_i * target_i // last_target  # Difficulty weight
        timespan += adj_time_i * (i - first)  # Recency weight
    timespan = timespan * 2 // (block_count + 1)  # Normalize recency weight
    target = last_target * timespan  # Standard retarget
    target //= TARGET_BLOCK_TIME * block_count
    return target_to_bits(target)


def next_bits_wt_compare(msg, block_count, limit_precision):
    with open("current_state.csv", 'w') as fh:
        for s in states:
            fh.write("%s,%s,%s\n" % (s.height, s.bits, s.timestamp))

    from subprocess import Popen, PIPE

    process = Popen(["./cashwork"], stdout=PIPE)
    (next_bits, err) = process.communicate()
    exit_code = process.wait()

    next_bits = int(next_bits.decode())
    next_bits_py = next_bits_wt(msg, block_count, limit_precision)
    if next_bits != next_bits_py:
        print("ERROR: Bits don't match. External %s, local %s" % (next_bits, next_bits_py))
        assert(next_bits == next_bits_py)
    return next_bits


def next_bits_dgw3(msg, block_count):
    ''' Dark Gravity Wave v3 from Dash '''
    block_reading = -1  # dito
    counted_blocks = 0
    last_block_time = 0
    actual_time_span = 0
    past_difficulty_avg = 0
    past_difficulty_avg_prev = 0
    i = 1
    while states[block_reading].height > 0:
        if i > block_count:
            break
        counted_blocks += 1
        if counted_blocks <= block_count:
            if counted_blocks == 1:
                past_difficulty_avg = bits_to_target(states[block_reading].bits)
            else:
                past_difficulty_avg = ((past_difficulty_avg_prev * counted_blocks) +
                                       bits_to_target(states[block_reading].bits)) // \
                                      (counted_blocks + 1)
        past_difficulty_avg_prev = past_difficulty_avg
        if last_block_time > 0:
            diff = last_block_time - states[block_reading].timestamp
            actual_time_span += diff
        last_block_time = states[block_reading].timestamp
        block_reading -= 1
        i += 1
    target_time_span = counted_blocks * TARGET_BLOCK_TIME
    target = past_difficulty_avg
    if actual_time_span < (target_time_span // 3):
        actual_time_span = target_time_span // 3
    if actual_time_span > (target_time_span * 3):
        actual_time_span = target_time_span * 3
    target = target // target_time_span
    target *= actual_time_span
    if target > MAX_TARGET:
        return MAX_BITS
    else:
        return target_to_bits(int(target))


def next_bits_current_dgw3(msg, block_count):
    ''' Dark Gravity Wave v3 from Dash as in current code'''
    block_reading = -1  # dito
    counted_blocks = 0
    past_difficulty_avg = 0
    past_difficulty_avg_prev = 0
    i = 1
    while states[block_reading].height > 0:
        if i > block_count:
            break
        counted_blocks += 1
        if counted_blocks <= block_count:
            if counted_blocks == 1:
                past_difficulty_avg = bits_to_target(states[block_reading].bits)
            else:
                past_difficulty_avg = ((past_difficulty_avg_prev * counted_blocks) +
                                       bits_to_target(states[block_reading].bits)) // \
                                      (counted_blocks + 1)
        past_difficulty_avg_prev = past_difficulty_avg
        block_reading -= 1
        i += 1
    target_time_span = block_count * TARGET_BLOCK_TIME
    actual_time_span = states[-1].timestamp - states[block_reading].timestamp
    if actual_time_span < (target_time_span // 3):
        actual_time_span = target_time_span // 3
    if actual_time_span > (target_time_span * 3):
        actual_time_span = target_time_span * 3
    target = past_difficulty_avg
    target = target // target_time_span
    target *= actual_time_span
    if target > MAX_TARGET:
        return MAX_BITS
    else:
        return target_to_bits(int(target))


def next_bits_fixed_dgw3(msg, block_count):
    ''' Fixed Dark Gravity Wave v3 from Dash '''
    block_reading = -1  # dito
    counted_blocks = 0
    past_difficulty_avg = 0
    past_difficulty_avg_prev = 0
    i = 1
    while states[block_reading].height > 0:
        if i > block_count:
            break
        counted_blocks += 1
        if counted_blocks <= block_count:
            if counted_blocks == 1:
                past_difficulty_avg = bits_to_target(states[block_reading].bits)
            else:
                past_difficulty_avg = ((past_difficulty_avg_prev * (counted_blocks - 1)) +
                                       bits_to_target(states[block_reading].bits)) // \
                                      (counted_blocks)
        past_difficulty_avg_prev = past_difficulty_avg
        block_reading -= 1
        i += 1
    target_time_span = block_count * TARGET_BLOCK_TIME
    actual_time_span = states[-1].timestamp - states[block_reading].timestamp
    if actual_time_span < (target_time_span // 3):
        actual_time_span = target_time_span // 3
    if actual_time_span > (target_time_span * 3):
        actual_time_span = target_time_span * 3
    target = past_difficulty_avg
    target = target // target_time_span
    target *= actual_time_span
    if target > MAX_TARGET:
        return MAX_BITS
    else:
        return target_to_bits(int(target))


def next_bits_xmr(msg, window):
    last_times = []
    last_difficulties = []
    for state in states[-window:]:
        last_times.append(state.timestamp)
        target = bits_to_target(state.bits)
        difficulty = TARGET_1 / target
        last_difficulties.append(difficulty)
    last_times = sorted(last_times)
    last_times = last_times[window // 6: -window // 6]
    last_difficulties = sorted(last_difficulties)
    last_difficulties = last_difficulties[window // 6: -window // 6]
    time_span = last_times[-1] - last_times[0]
    if time_span == 0:
        time_span = 1
    diff_sum = sum(last_difficulties)
    result_difficulty = (diff_sum * TARGET_BLOCK_TIME + time_span + 1) // time_span
    result_target = int(TARGET_1 // result_difficulty)
    if result_target > MAX_TARGET:
        return MAX_BITS
    return target_to_bits(result_target)


def next_bits_cdho(msg):
    last_block_time1 = states[-1].timestamp - states[-2].timestamp
    last_block_time2 = states[-2].timestamp - states[-3].timestamp
    last_block_time3 = states[-3].timestamp - states[-4].timestamp
    last_block_time4 = states[-4].timestamp - states[-5].timestamp
    last_block_time5 = states[-5].timestamp - states[-6].timestamp

    error1 = last_block_time1 - TARGET_BLOCK_TIME
    error2 = last_block_time2 - TARGET_BLOCK_TIME
    error3 = last_block_time3 - TARGET_BLOCK_TIME
    error4 = last_block_time4 - TARGET_BLOCK_TIME
    error5 = last_block_time5 - TARGET_BLOCK_TIME

    error_der1 = (error1 - error3) // (states[-1].wall_time - states[-3].wall_time)
    error_der2 = (error2 - error4) // (states[-2].wall_time - states[-4].wall_time)
    error_der3 = (error3 - error5) // (states[-3].wall_time - states[-5].wall_time)

    error_second_der = (error_der1 - error_der3) // (states[-2].wall_time - states[-4].wall_time)

    D = error_der2 * error_der2 - error3 * error_second_der

    if error3 and (D > 0):
        omega = (-error_der2 + math.sqrt(D)) // error3
    else:
        omega = 0

    # common solution e = (a + b * t) * exp(-omega*t)
    a = error3
    b = error_der2 + omega * error3

    # find next value
    if omega > 0:
        next_t = states[-1].wall_time + TARGET_BLOCK_TIME
        next_error = (a + b * next_t) * math.exp(-1 * omega * next_t)
        target_time = TARGET_BLOCK_TIME + next_error
        if target_time < 1:
            target_time = 1
    else:
        target_time = TARGET_BLOCK_TIME

    prev_target = bits_to_target(states[-1].bits)

    if last_block_time1 > 10 * TARGET_BLOCK_TIME:
        target_time = TARGET_BLOCK_TIME

    k = last_block_time1 / target_time

    if k < 0.7:
        k = 0.7
    if k > 1.3:
        k = 1.3

    result_target = int(prev_target * k)
    if result_target > MAX_TARGET:
        return MAX_BITS
    return target_to_bits(result_target)


class PidController:
    def __init__(self):
        self.bits = INITIAL_DASH_BITS

    def update(self):
        self.bits = states[-1].bits

    def get_retarget_output(self):
        return self.bits


pid_controller = PidController()


def next_bits_pid(msg):
    pid_controller.update()
    return pid_controller.get_retarget_output()


def next_bits_simple_align(msg):
    last_block_time = states[-1].timestamp - states[-2].timestamp
    if last_block_time < 1:
        last_block_time = 1
    target_time = (2 * TARGET_BLOCK_TIME) - last_block_time
    if target_time < 1:
        target_time = 1
    prev_target = bits_to_target(states[-1].bits)
    k = last_block_time / target_time

    if k < 0.7:
        k = 0.7
    if k > 1.3:
        k = 1.3

    result_target = int(prev_target * k)
    if result_target > MAX_TARGET:
        return MAX_BITS
    return target_to_bits(result_target)


def next_bits_proportional_error(msg, window):
    last_times = []
    for state in states[-window:]:
        last_times.append(state.timestamp)
    last_times = sorted(last_times)
    last_block_time = (last_times[-1] - last_times[0]) / (window - 1)

    if last_block_time < 1:
        last_block_time = 1
    target_time = (TARGET_BLOCK_TIME + last_block_time) // 2
    if target_time < 1:
        target_time = 1
    prev_target = bits_to_target(states[-1].bits)
    k = last_block_time / target_time
    k = (k - 1) / 2 + 1

    result_target = int(prev_target * k)
    if result_target > MAX_TARGET:
        return MAX_BITS
    return target_to_bits(result_target)


def next_bits_m2(msg, window_1, window_2):
    interval_target = compute_target(-1 - window_1, -1)
    interval_target += compute_target(-2 - window_2, -2)
    return target_to_bits(interval_target >> 1)


def next_bits_m4(msg, window_1, window_2, window_3, window_4):
    interval_target = compute_target(-1 - window_1, -1)
    interval_target += compute_target(-2 - window_2, -2)
    interval_target += compute_target(-3 - window_3, -3)
    interval_target += compute_target(-4 - window_4, -4)
    return target_to_bits(interval_target >> 2)


def block_time(mean_time):
    # Sample the exponential distn
    sample = random.random()
    lmbda = 1 / mean_time
    res = math.log(1 - sample) / -lmbda
    if res < 1:
        res = 1
    return res


def next_step(algo, scenario):
    # First figure out our hashrate
    msg = []
    hashrate = scenario.hashrate(msg, **scenario.params)
    # Calculate our dynamic difficulty
    bits = algo.next_bits(msg, **algo.params)
    target = bits_to_target(bits)
    # See how long we take to mine a block
    mean_hashes = pow(2, 256) // target
    mean_time = mean_hashes / (hashrate * 1e15)
    time = int(block_time(mean_time) + 0.5)
    wall_time = states[-1].wall_time + time
    # Did the difficulty ramp hashrate get the block?
#    if random.random() < (scenario.dr_hashrate / hashrate):
#        timestamp = median_time_past(states[-11:]) + 1
#    else:
    timestamp = wall_time

    chainwork = states[-1].chainwork + bits_to_work(bits)

    # add a state
    states.append(State(states[-1].height + 1, wall_time, timestamp,
                        bits, chainwork, hashrate, ' / '.join(msg)))


Algo = namedtuple('Algo', 'next_bits params')

Algos = {
    'k-1': Algo(next_bits_k, {
        'mtp_window': 6,
        'high_barrier': 60 * 128,
        'target_raise_frac': 64,   # Reduce difficulty ~ 1.6%
        'low_barrier': 60 * 30,
        'target_drop_frac': 256,   # Raise difficulty ~ 0.4%
        'fast_blocks_pct': 95,
    }),
    'k-2': Algo(next_bits_k, {
        'mtp_window': 4,
        'high_barrier': 60 * 55,
        'target_raise_frac': 100,   # Reduce difficulty ~ 1.0%
        'low_barrier': 60 * 36,
        'target_drop_frac': 256,   # Raise difficulty ~ 0.4%
        'fast_blocks_pct': 95,
    }),
    'd-1': Algo(next_bits_d, {}),
    'cw-72': Algo(next_bits_cw, {
        'block_count': 72,
    }),
    'cw-108': Algo(next_bits_cw, {
        'block_count': 108,
    }),
    'cw-144': Algo(next_bits_cw, {
        'block_count': 144,
    }),
    'cw-sha-16': Algo(next_bits_sha, {}),
    'cw-180': Algo(next_bits_cw, {
        'block_count': 180,
    }),
    'wt-144': Algo(next_bits_wt, {
        'block_count': 144,
        'limit_precision': False
    }),
    'dgw3-24': Algo(next_bits_dgw3, {  # 24-blocks, like Dash
        'block_count': 24,
    }),
    'cdgw3-24': Algo(next_bits_current_dgw3, {  # 24-blocks, like Dash
        'block_count': 24,
    }),
    'fdgw3-24': Algo(next_bits_fixed_dgw3, {  # 24-blocks, like Dash
        'block_count': 24,
    }),
    'dgw3-144': Algo(next_bits_dgw3, {  # 1 full day
        'block_count': 144,
    }),
    'cdgw3-144': Algo(next_bits_current_dgw3, {  # 1 full day
        'block_count': 144,
    }),
    'fdgw3-144': Algo(next_bits_fixed_dgw3, {  # 1 full day
        'block_count': 144,
    }),
    'xmr': Algo(next_bits_xmr, {
       'window': 720
    }),
    'cdho': Algo(next_bits_cdho, {
    }),
    'pid': Algo(next_bits_pid, {
    }),
    'sa': Algo(next_bits_simple_align, {
    }),
    'pe': Algo(next_bits_proportional_error, {
        'window': 10
    }),
    'meng-1': Algo(next_bits_m2, {  # mengerian_algo_1
        'window_1': 71,
        'window_2': 137,
    }),
    'meng-2': Algo(next_bits_m4, {  # mengerian_algo_2
        'window_1': 13,
        'window_2': 37,
        'window_3': 71,
        'window_4': 137,
    }),
    # runs wt-144 in external program, compares with python implementation.
    'wt-144-compare': Algo(next_bits_wt_compare, {
        'block_count': 144,
        'limit_precision': True
    })
}


def const_hashrate(msg, base_rate):
    return base_rate


def random_oscillations_hashrate(msg, base_rate,  amplitude):
    return base_rate * (1 + amplitude * (random.random() - 0.5))


def inout_hashrate(msg, base_rate, additional_rate):
    height = len(states)
    if(height // 100) % 2:
        return base_rate
    else:
        return base_rate + additional_rate


def fake_ts_hashrate(msg, base_rate):
    return base_rate


Scenario = namedtuple('Scenario', 'hashrate params')

Scenarios = {
    'const': Scenario(const_hashrate, {
        'base_rate': INITIAL_HASHRATE
    }),
    'random': Scenario(random_oscillations_hashrate, {
        'base_rate': INITIAL_HASHRATE,
        'amplitude': 0.1
    }),
    'increase': Scenario(const_hashrate, {
        'base_rate':  2 * INITIAL_HASHRATE
    }),
    'decrease': Scenario(const_hashrate, {
        'base_rate': 0.5 * INITIAL_HASHRATE
    }),
    'inout': Scenario(inout_hashrate, {
        'base_rate': INITIAL_HASHRATE,
        'additional_rate': INITIAL_HASHRATE
    }),
    'fake_timestamp': Scenario(fake_ts_hashrate, {
        'base_rate': INITIAL_HASHRATE
    })
}


def run_one_simul(algo, scenario, print_it):
    states.clear()

    # Initial state is afer 2020 steady prefix blocks
    N = 2020
    for n in range(-N, 0):
        state = State(INITIAL_HEIGHT + n, INITIAL_TIMESTAMP + n * TARGET_BLOCK_TIME,
                      INITIAL_TIMESTAMP + n * TARGET_BLOCK_TIME,
                      INITIAL_DASH_BITS, INITIAL_SINGLE_WORK * (n + N + 1),
                      INITIAL_HASHRATE, '')
        states.append(state)

    # Run the simulation
    if print_it:
        print_headers()
    for n in range(10000):
        next_step(algo, scenario)
        if print_it:
            print_state()

    # Drop the prefix blocks to be left with the simulation blocks
    simul = states[N:]

    block_times = [simul[n + 1].timestamp - simul[n].timestamp
                   for n in range(len(simul) - 1)]
    return block_times


def check_random_times():
    times = []
    for i in range(10000):
        times.append(block_time(TARGET_BLOCK_TIME))

    print("mean=%f" % statistics.mean(times))
    print("std_dev=%f" % statistics.stdev(times))
    print("median = %f" % (sorted(times)[len(times) // 2]))
    print("max = %f" % max(times))


def main():
    '''Outputs CSV data to stdout.   Final stats to stderr.'''

    parser = argparse.ArgumentParser('Run a mining simulation')
    parser.add_argument('-a', '--algo', metavar='algo', type=str,
                        choices=list(Algos.keys()),
                        default='pe', help='algorithm choice')
    parser.add_argument('-s', '--scenario', metavar='scenario', type=str,
                        choices=list(Scenarios.keys()),
                        default='const', help='scenario choice')
    parser.add_argument('-r', '--seed', metavar='seed', type=int,
                        default=None, help='random seed')
    parser.add_argument('-n', '--count', metavar='count', type=int,
                        default=1, help='count of simuls to run')
    args = parser.parse_args()

    count = max(1, args.count)
    algo = Algos.get(args.algo)
    scenario = Scenarios.get(args.scenario)
    seed = int(time.time()) if args.seed is None else args.seed

    print("Algo %s,  scenario %s" % (args.algo, args.scenario))

    to_stderr = partial(print, file=sys.stderr)
    to_stderr("Starting seed {} for {} simuls".format(seed, count))

    means = []
    std_devs = []
    medians = []
    maxs = []
    for loop in range(count):
        random.seed(seed)
        seed += 1
        block_times = run_one_simul(algo, scenario, count == 1)
        means.append(statistics.mean(block_times))
        std_devs.append(statistics.stdev(block_times))
        medians.append(sorted(block_times)[len(block_times) // 2])
        maxs.append(max(block_times))

    def stats(text, values):
        if count == 1:
            to_stderr('{} {}s'.format(text, values[0]))
        else:
            to_stderr('{}(s) Range {:0.1f}-{:0.1f} Mean {:0.1f} '
                      'Std Dev {:0.1f} Median {:0.1f}'
                      .format(text, min(values), max(values),
                              statistics.mean(values),
                              statistics.stdev(values),
                              sorted(values)[len(values) // 2]))

    stats("Mean   block time", means)
    stats("StdDev block time", std_devs)
    stats("Median block time", medians)
    stats("Max    block time", maxs)


if __name__ == '__main__':
    main()
