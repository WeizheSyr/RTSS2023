import goto
from goto import with_goto
from goto import goto, label


@with_goto  # 必须有，且添加在你要使用goto函数前
def test(list_):
    tmp_list = list_
    label.begin  # 标识跳转并开始执行的地方
    result = []
    try:
        for i, j in enumerate(list_):
            tmp = 1 / j
            result.append(tmp)
            last_right_i = i
    except ZeroDivisionError:
        del tmp_list[last_right_i + 1]
        goto.begin  # 在有跳转标识的地方开始执行
    return result


a = test([1, 3, 4, 0, 6])
print(a)