def convert_and_prefix_file(input_file_path, output_file_path, start_index):
    try:
        with open(input_file_path, "r") as file:
            content = file.read()

        # 检查每行是否确实有256个字符
        lines = content.splitlines()
        for line in lines:
            if len(line) != 256:
                raise ValueError(
                    "File does not contain lines of exactly 256 characters."
                )

        # 将每行256个字符的文本转换为每行32个字符，并添加前缀
        new_content = ""
        for i, line in enumerate(lines):
            for j in range(0, 256, 32):
                index = start_index + i * 8 + j // 32
                new_content += f"CMEM[{index}]=0b{line[j:j+32]};\n"

        # 写入新的文件
        with open(output_file_path, "w") as file:
            file.write(new_content)

        print(
            f"File '{input_file_path}' has been converted and saved as '{output_file_path}'."
        )

    except FileNotFoundError:
        print(f"The file '{input_file_path}' does not exist.")
    except ValueError as e:
        print(e)
    except Exception as e:
        print(f"An error occurred: {e}")


# 使用示例
input_path = "wet_of_cnn.txt"  # 替换为你的输入文件路径
output_path = "wet_output.txt"  # 替换为你的输出文件路径
start_index = 3200 + 32768 * 3 - 1152 * 3  # 替换为你希望开始的索引号
convert_and_prefix_file(input_path, output_path, start_index)
