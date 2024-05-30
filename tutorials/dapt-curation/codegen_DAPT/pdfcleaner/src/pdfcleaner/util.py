def is_word(struct):
    """Checks if a structure is a word"""
    return isinstance(struct, tuple) and isinstance(struct[0], str)


def word_count(struct):
    """Counts the number of words inside a page/flow/block/line"""
    if is_word(struct):
        return 1
    return sum(word_count(x) for x in struct[0])


def get_words(struct):
    """Returns the raw words inside a page/flow/block/line"""
    if is_word(struct):
        return [struct[0]]
    return [x for y in struct[0] for x in get_words(y)]


def struct_depth(struct):
    """Given a structure, returns the depth of the structure"""
    if is_word(struct):
        return 0
    return 1 + struct_depth(struct[0][0])


def sort_struct(struct, multi_col=True):
    """Given a structure, sorts the sub-structures by natural reading order

    Natural reading order has the following rules:
     - Initialize an empty stack of collections and and empty list to sort into
     - Create a collection of all structures and push it onto the stack
     - While the stack is not empty:
        - Pop the top collection off the stack
        - If the collection does not contain any non-"visited" structures,
        continue
        - Take the top-left structure from the collection
            - Push this into the sorted list
            - Mark this structure as "visited"
        - Push the original collection back onto the stack
        - Create a new collection, consisting of all structures from the
        original collection that have any X-axis overlap with the chosen
        structure. Push this new collection onto the stack.
    """
    stack = [struct[0]]
    pos = struct[1]
    ret = []
    unvisited = set([x[1] for x in struct[0]])
    top_left = None
    while stack:
        group = stack.pop()
        # Filter out visited structures
        group = [x for x in group if x[1] in unvisited]
        # If group is empty, clear the current top_left
        if not group:
            top_left = None
            continue
        # Push the original group back onto the stack
        stack.append(group)

        if multi_col:
            # Find the left-most structure
            left = min(group, key=lambda x: x[1][0])
            # Create a new group of structures that are on the same "column"
            new_group = [
                x for x in group if x[1][0] <= left[1][2] and x[1][2] >= left[1][0]
            ]
            # Find the top structure in this group
            top_left = min(new_group, key=lambda x: x[1][1])
            # Create a new group of structures that are on the same "column"
            new_group = [
                x
                for x in struct[0]
                if x[1] in unvisited
                and x[1][0] <= top_left[1][2]
                and x[1][2] >= top_left[1][0]
            ]
            # Find the top structure in this group
            top_left = min(new_group, key=lambda x: x[1][1])
        else:
            # Find the top-most structure
            top = min(group, key=lambda x: x[1][1])
            # Create a new group of structures that are on the same "row"
            new_group = [
                x for x in group if x[1][1] <= top[1][3] and x[1][3] >= top[1][1]
            ]
            # Find the left structure in this group
            top_left = min(new_group, key=lambda x: x[1][0])
            # Create a new group of structures that are on the same "row"
            new_group = [
                x
                for x in struct[0]
                if x[1] in unvisited
                and x[1][1] <= top_left[1][3]
                and x[1][3] >= top_left[1][1]
            ]
            # Find the left structure in this group
            top_left = min(new_group, key=lambda x: x[1][0])

        # Add top_left to sorted list
        ret.append(top_left)
        unvisited.remove(top_left[1])
        # Push new group onto stack
        stack.append(new_group)
    return (ret, pos)


def merge_structs(structs):
    """Given a list of structs, merge them all into the first one"""
    # Need to do this in-place
    while len(structs) > 1:
        structs[0][0].extend(structs.pop(1)[0])


def merge_parallel_structs(structs):
    """Given a list of structs, merge all structs that share the first's y-pos"""
    # Need to do this in-place
    idx = 1
    target = 0
    found = False
    while len(structs) > idx:
        payload = structs[idx]
        if structs[target][1][1] == payload[1][1]:
            structs[target][0].extend(payload[0])
            structs.pop(idx)
            found = True
        else:
            idx += 1
            if not found:
                target += 1
    if found:
        merge_parallel_structs(structs)


def strip_pos(struct):
    """Strips a structure of all position information"""
    if is_word(struct):
        return struct[0]
    return [strip_pos(x) for x in struct[0]]


def dehyphenate_page(page):
    """Given a posless structure, removes end-of-line hyphenation

    This occurs every time the line-end word ends with a hyphen.
    NOTE: This is incorrect for words that are actually hyphenated.

    This is done by taking the line-end word and moving it to the next struct.

    This method assumes that the struct has already been sorted into
    natural reading order and then parsed down to just the words.
    """
    for f, flow in enumerate(page):
        for b, block in enumerate(flow):
            for l, line in enumerate(block):
                word = line[-1]
                if not word:
                    continue
                if not word[-1] in ["-", "‐", "–"]:
                    continue
                if len(word) > 1 and word[-2] in ["-", "‐", "–"]:
                    continue
                target_line = None
                if l < len(block) - 1:
                    target_line = block[l + 1]
                elif b < len(flow) - 1:
                    target_line = flow[b + 1][0]
                elif f < len(page) - 1:
                    target_line = page[f + 1][0][0]
                if target_line is None:
                    continue
                target_line[0] = word[:-1] + target_line[0]
                if True:
                    line.pop()
                if len(line) == 0:
                    block.pop(l)
                if len(block) == 0:
                    flow.pop(b)
                if len(flow) == 0:
                    page.pop(f)
    return page


def dump_text(posless):
    """Given a posless structure, dumps the text out.

    Words are separated by spaces.
    Lines are separated by 1 new-line character.
    Blocks, flows, and pages are separated by 2 new-line characters.
    """
    if type(posless) is str:
        return posless
    if type(posless[0]) is str:
        return " ".join(dump_text(x) for x in posless)
    #    return "\n".join(dump_text(x) for x in posless)
    if type(posless[0][0]) is str:
        return "\n".join(dump_text(x) for x in posless)
    return "\n\n".join(dump_text(x) for x in posless)


if __name__ == "__main__":
    raise RuntimeError("This file is importable, but not executable")
