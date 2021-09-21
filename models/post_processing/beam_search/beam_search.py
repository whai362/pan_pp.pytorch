import torch

from .topk import TopK


class BeamNode(object):
    def __init__(self, seq, state, score):
        self.seq = seq
        self.state = state
        self.score = score
        self.avg_score = score / len(seq)

    def __cmp__(self, other):
        if self.avg_score == other.avg_score:
            return 0
        elif self.avg_score < other.avg_score:
            return -1
        else:
            return 1

    def __lt__(self, other):
        return self.avg_score < other.avg_score

    def __eq__(self, other):
        return self.avg_score == other.avg_score


class BeamSearch(object):
    """Class to generate sequences from an image-to-text model."""
    def __init__(self, decode_step, eos, beam_size=2, max_seq_len=32):
        self.decode_step = decode_step
        self.eos = eos
        self.beam_size = beam_size
        self.max_seq_len = max_seq_len

    def beam_search(self, init_inputs, init_states):
        # self.beam_size = 1
        batch_size = len(init_inputs)
        part_seqs = [TopK(self.beam_size) for _ in range(batch_size)]
        comp_seqs = [TopK(self.beam_size) for _ in range(batch_size)]

        # print(init_inputs.shape, init_states.shape)
        words, scores, states = self.decode_step(init_inputs,
                                                 init_states,
                                                 k=self.beam_size)
        for batch_id in range(batch_size):
            for i in range(self.beam_size):
                node = BeamNode([words[batch_id][i]],
                                states[:, :, batch_id, :], scores[batch_id][i])
                part_seqs[batch_id].push(node)

        for t in range(self.max_seq_len - 1):
            part_seq_list = []
            for p in part_seqs:
                part_seq_list.append(p.extract())
                p.reset()

            inputs, states = [], []
            for seq_list in part_seq_list:
                for node in seq_list:
                    inputs.append(node.seq[-1])
                    states.append(node.state)
            if len(inputs) == 0:
                break

            inputs = torch.stack(inputs)
            states = torch.stack(states, dim=2)
            words, scores, states = self.decode_step(inputs,
                                                     states,
                                                     k=self.beam_size + 1)

            idx = 0
            for batch_id in range(batch_size):
                for node in part_seq_list[batch_id]:
                    tmp_state = states[:, :, idx, :]
                    k = 0
                    num_hyp = 0
                    while num_hyp < self.beam_size:
                        word = words[idx][k]
                        tmp_seq = node.seq + [word]
                        tmp_score = node.score + scores[idx][k]
                        tmp_node = BeamNode(tmp_seq, tmp_state, tmp_score)
                        k += 1
                        num_hyp += 1

                        if word == self.eos:
                            comp_seqs[batch_id].push(tmp_node)
                            num_hyp -= 1
                        else:
                            part_seqs[batch_id].push(tmp_node)
                    idx += 1

        for batch_id in range(batch_size):
            if not comp_seqs[batch_id].size():
                comp_seqs[batch_id] = part_seqs[batch_id]
        seqs = [seq_list.extract(sort=True)[0].seq for seq_list in comp_seqs]
        seq_scores = [
            seq_list.extract(sort=True)[0].avg_score for seq_list in comp_seqs
        ]
        return seqs, seq_scores
