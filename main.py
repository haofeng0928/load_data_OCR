from data_gen import DataGen


def main():
    data_root_dir = r'D:\myData\huawei_datetext\train_img'
    data_path = r'D:\myData\huawei_datetext\train_txt.txt'
    lexicon_file = 'date_lexicon.txt'

    gens = DataGen(data_root_dir, data_path, lexicon_file=lexicon_file, mean=[128], channel=1,
                   evaluate=False, valid_target_len=float('inf'))

    for i, batch in enumerate(gens.gen(32)):
        print("Child Traing batch: " + str(i))

    print(gens.min_list, gens.max_list)


if __name__ == '__main__':
    main()

