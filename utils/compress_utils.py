import lzma
import dahuffman
import pickle
import numpy as np
import torch

def huffman_encode(data):
    codec = dahuffman.HuffmanCodec.from_data(data)
    encoded_bytes = codec.encode(data)
    huffman_table = codec.get_code_table()
    return encoded_bytes, huffman_table

def huffman_decode(encoded_bytes, huffman_table):
    codec = dahuffman.HuffmanCodec(code_table=huffman_table)
    decoded_data = codec.decode(encoded_bytes)
    return np.array(decoded_data, dtype=np.uint16)

def save_comp(filename, save_dict):
    with lzma.open(filename, "wb") as f:
        pickle.dump(save_dict, f)

def load_comp(filename):
    with lzma.open(filename, "rb") as f:
        save_dict = pickle.load(f)
    return save_dict

def write_storage(save_dict, byte, numG):
    for name in save_dict:
        if name == 'xyz':
            byte['xyz'] = len(save_dict['xyz'])
        elif 'MLP' in name:
            byte['MLPs'] += save_dict[name].shape[0]*16/8
        else:
            attr, comp = name.split('_')
            if 'code' in comp:
                for i in range(len(save_dict[name])):
                    byte[attr] += save_dict[name][i].shape[0]*save_dict[name][i].shape[1]*16/8
            else:
                for i in range(len(save_dict[name])):
                    byte[attr] += len(save_dict[name][i])
    byte['total'] = byte['xyz'] + byte['scale'] + byte['rotation'] + byte['app'] + byte['MLPs']
    return "#G: " + str(numG) + "\nPosition: " + str(byte['xyz']) + "\nScale: " + str(byte['scale']) + "\nRotation: " + str(byte['rotation']) + "\nAppearance: " + str(byte['app']) + "\nMLPs: " + str(byte['MLPs'])+ "\nTotal: " + str(byte['total']) + "\n"

def splitBy3(a):
    x = a & 0x1FFFFF
    x = (x | x << 32) & 0x1F00000000FFFF
    x = (x | x << 16) & 0x1F0000FF0000FF
    x = (x | x << 8) & 0x100F00F00F00F00F
    x = (x | x << 4) & 0x10C30C30C30C30C3
    x = (x | x << 2) & 0x1249249249249249
    return x


def mortonEncode(pos: torch.Tensor) -> torch.Tensor:
    x, y, z = pos.unbind(-1)
    answer = torch.zeros(len(pos), dtype=torch.long, device=pos.device)
    answer |= splitBy3(x) | splitBy3(y) << 1 | splitBy3(z) << 2
    return answer