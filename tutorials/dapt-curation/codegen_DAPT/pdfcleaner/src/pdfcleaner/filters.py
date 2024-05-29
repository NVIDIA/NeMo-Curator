from .util import word_count, get_words, merge_parallel_structs
import re

def manuals_filter(pages, err="errors.log"):
    """Filter out unwanted text from manuals"""
    short_limit = 10
    medium_limit = 10
    long_limit = 25
    consec_limit = 5
    first_page = True
    err_out = open(err, "a+")
    for page in pages:
        try:
            skip_page = False
            num_flows = len(page[0])
            is_short = [False] * num_flows
            is_long = [False] * num_flows
            is_consec_int = [False] * num_flows
            delete_me = [False] * num_flows
            bad_section = [False] * num_flows
            bullets = [False] * num_flows
            figure_caption = []
            table_caption = []
            last_int = -1
            for idx, flow in enumerate(page[0]):
                word_list = [w.lower() for w in get_words(flow)]
                word_dump = " ".join(word_list)

                if "table of contents" in word_dump \
                            or "index" in word_dump:
                    bad_section[idx] = True

                # Detect consecutive integers
                if word_dump.isdigit():
                    try:
                        next_int = int(word_dump)
                        if next_int == last_int + 1:
                            is_consec_int[idx] = True
                            if all(is_consec_int[idx-consec_limit:]):
                                delete_me[idx-consec_limit:] = [True]*(consec_limit+1)
                        last_int = next_int
                    except:
                        pass

                # Detect abnormally short flows
                flag = True
                for block in flow[0]:
                    block_wc = word_count(block)
                    if block_wc > short_limit:
                        flag = False
                        break
                is_short[idx] = flag

                # Detect long flows
                if len(word_list) > long_limit:
                    is_long[idx] = True

                if len(word_list) > 1:
                    # Detect figure captions
                    if word_list[0] in ["figure"] and word_list[1][-1] == ":":
                        delete_me[idx] = True
                        is_short[idx] = True
                        figure_caption.append(idx)
                        continue
                    # Detect table captions
                    if word_list[0] in ["table"] and word_list[1][-1] == ":":
                        table_caption.append(idx)

                # Detect copyright disclaimers
                if first_page:
                    if "©" in word_dump and "isbn" in word_dump:
                        delete_me[idx] = True
                        continue
                    if "permission to make digital or hard copies" in word_dump:
                        delete_me[idx] = True
                        continue

                # Detect loose bullet points
                if word_list[-1] in ["•", "●", "o", "❍"] \
                        and len(get_words(flow[0][-1])) == 1:
                    bullets[idx] = True
                    if len(word_list) == 1:
                        delete_me[idx] = True
                    continue

                # Merge parallel lines inside blocks
                for block in flow[0]:
                    merge_parallel_structs(block[0])
                # Merge parallel blocks
                merge_parallel_structs(flow[0])

            # Iterate over all figure captions found
            for i in figure_caption:
                # Use the index to detect unbroken chains of short flows
                forward_chain = i
                backward_chain = i
                while forward_chain < num_flows-1 and is_short[forward_chain]:
                    forward_chain += 1
                while backward_chain > 0 and is_short[backward_chain]:
                    backward_chain -= 1
                # Delete the longer chain
                if forward_chain - i > i - backward_chain:
                    for j in range(i, forward_chain):
                        delete_me[j] = True
                else:
                    for j in range(backward_chain, i):
                        delete_me[j] = True

            # Iterate over all table captions found
            extracted_tables = {}
            for i in table_caption:
                # Use the index to detect unbroken chains of short flows
                forward_chain = i
                backward_chain = i
                while forward_chain < num_flows-1 and is_short[forward_chain]:
                    forward_chain += 1
                while backward_chain > 0 and is_short[backward_chain]:
                    backward_chain -= 1

                # Extract the longer chain
                extracted_table = []
                extracted_table.append(page[0][i])
                delete_me[i] = True
                if forward_chain - i > i - backward_chain:
                    for j in range(i+1, forward_chain):
                        extracted_table.append(page[0][j])
                        delete_me[j] = True
                else:
                    for j in range(backward_chain, i):
                        extracted_table.append(page[0][j])
                        delete_me[j] = True
                # Find the next section_start
                next_section = len(page[0])
                # Add the extracted table to the next section
                if next_section in extracted_tables:
                    extracted_tables[next_section].extend(extracted_table)
                else:
                    extracted_tables[next_section] = extracted_table

            # Get miny / maxy for all long flows
            miny = page[1][1]*0.05
            maxy = page[1][1]*0.90

            # Use these miny to see if we're in a bad section
            for idx, flow in enumerate(page[0]):
                if flow[1][1] <= miny and bad_section[idx]:
                    skip_page = True

            # Use these miny / maxy to delete headers/footers
            for idx, flow in enumerate(page[0]):
                if flow[1][1] < miny and not first_page \
                        and idx not in table_caption:
                    delete_me[idx] = True
                if flow[1][3] > maxy:
                    delete_me[idx] = True

            # Merge bullet points
            for idx, flow in enumerate(page[0]):
                if bullets[idx] and idx+1 < len(page[0]):
                    # Retrieve original bullet point block
                    orig_block = flow[0].pop(-1)
                    orig_word = orig_block[0][0]\
                                          [0][0]
                    # Insert it into the next flow's first word
                    next_flow = page[0][idx+1]
                    next_block = next_flow[0][0]
                    next_line = next_block[0][0]
                    next_line[0].insert(0, orig_word)

            # Delete all flows marked with delete_me
            # Add in extracted tables
            new_page = ([], page[1])
            for i, flow in enumerate(page[0]):
                if i in extracted_tables:
                    l = len(new_page[0])-1
                    new_page[0][l:l] = extracted_tables[i]
                if not delete_me[i]:
                    new_page[0].append(flow)

            if first_page:
                first_page = False

            # Filter out abnormally short pages
            page_wc = word_count(new_page)
            if page_wc < medium_limit:
                skip_page = True

            if skip_page:
                continue

            yield new_page

        except Exception as e:
            print(f"Error {e} in page\n{page}", file=err_out)


def academic_filter(pages, err="errors.log"):
    """Filter out unwanted text from academic work"""
    short_limit = 10
    medium_limit = 15
    long_limit = 25
    consec_limit = 5
    section_titles = ["abstract", "introduction", "conclusion", "references",
            "acknowledgements", "problem", "evaluation", "method", "methods",
            "results", "prior", "related", "discussion"]
    first_page = True
    err_out = open(err, "a+")
    for page in pages:
        try:
            num_flows = len(page[0])
            is_short = [False] * num_flows
            is_long = [False] * num_flows
            is_consec_int = [False] * num_flows
            delete_me = [False] * num_flows
            figure_caption = []
            table_caption = []
            section_start = []
            ref_section = False
            last_int = -1
            for idx, flow in enumerate(page[0]):
                word_list = [w.lower() for w in get_words(flow)]
                word_dump = " ".join(word_list)

                if word_dump.isdigit():
                    try:
                        next_int = int(word_dump)
                        if next_int == last_int + 1:
                            is_consec_int[idx] = True
                            if all(is_consec_int[idx-consec_limit:]):
                                delete_me[idx-consec_limit:] = [True]*(consec_limit+1)
                        last_int = next_int
                    except:
                        pass

                # Detect abnormally short flows
                flag = True
                for block in flow[0]:
                    block_wc = word_count(block)
                    if block_wc > short_limit:
                        flag = False
                        break
                # Don't count section titles
                for st in section_titles:
                    if st in word_list:
                        flag=False
                        section_start.append(idx)
                        break
                is_short[idx] = flag

                # Detect long flows
                if len(word_list) > long_limit:
                    is_long[idx] = True

                if len(word_list) > 1:
                    # Detect figure captions
                    if word_list[0] in ["figure"] and word_list[1][-1] == ":":
                        delete_me[idx] = True
                        is_short[idx] = True
                        figure_caption.append(idx)
                        continue
                    # Detect table captions
                    if word_list[0] in ["table"] and word_list[1][-1] == ":":
                        table_caption.append(idx)
                # Detect References section
                if "references" in word_list:
                    ref_section = True
                    delete_me[idx] = True
                if ref_section and bool(re.search(r'\[\d+\]', word_list[0])):
                    to_delete = True
                    delete_me[idx] = True

                # Detect copyright disclaimers
                if first_page:
                    if "©" in word_dump and "isbn" in word_dump:
                        delete_me[idx] = True
                        continue
                    if "work is supported by" in word_dump:
                        delete_me[idx] = True
                        continue
                    if "permission to make digital or hard copies" in word_dump:
                        delete_me[idx] = True
                        continue
                    if "reference format" in word_dump:
                        delete_me[idx] = True
                        continue
                    if "doi.org" in word_dump:
                        delete_me[idx] = True
                        continue

            # Iterate over all figure captions found
            for i in figure_caption:
                # Use the index to detect unbroken chains of short flows
                forward_chain = i
                backward_chain = i
                while forward_chain < num_flows-1 and is_short[forward_chain]:
                    forward_chain += 1
                while backward_chain > 0 and is_short[backward_chain]:
                    backward_chain -= 1
                # Delete the longer chain
                if forward_chain - i > i - backward_chain:
                    for j in range(i, forward_chain):
                        delete_me[j] = True
                else:
                    for j in range(backward_chain, i):
                        delete_me[j] = True

            # Iterate over all table captions found
            extracted_tables = {}
            for i in table_caption:
                # Use the index to detect unbroken chains of short flows
                forward_chain = i
                backward_chain = i
                while forward_chain < num_flows-1 and is_short[forward_chain]:
                    forward_chain += 1
                while backward_chain > 0 and is_short[backward_chain]:
                    backward_chain -= 1

                # Extract the longer chain
                extracted_table = []
                extracted_table.append(page[0][i])
                delete_me[i] = True
                if forward_chain - i > i - backward_chain:
                    for j in range(i+1, forward_chain):
                        extracted_table.append(page[0][j])
                        delete_me[j] = True
                else:
                    for j in range(backward_chain, i):
                        extracted_table.append(page[0][j])
                        delete_me[j] = True
                # Find the next section_start
                next_section = len(page[0])
                prev_section = 0
                for j in section_start:
                    if j > i:
                        next_section = j
                        break
                    prev_section = j
                if next_section == len(page[0]):
                    next_section = prev_section
                # Add the extracted table to the next section
                if next_section in extracted_tables:
                    extracted_tables[next_section].extend(extracted_table)
                else:
                    extracted_tables[next_section] = extracted_table

            # Get miny / maxy for all long flows
            miny = page[1][1]
            maxy = 0
            for idx, flow in enumerate(page[0]):
                if not is_long[idx]:
                    continue
                if flow[1][1] < miny:
                    miny = flow[1][1]
                if flow[1][3] > maxy:
                    maxy = flow[1][3]
            if maxy == 0:
                maxy = page[1][1]
            if miny == page[1][1]:
                miny = 0
            # Use these miny / maxy to delete headers/footers
            for idx, flow in enumerate(page[0]):
                if flow[1][1] < miny-15 and not first_page \
                        and idx not in table_caption:
                    delete_me[idx] = True
                if flow[1][3] > maxy+15:
                    delete_me[idx] = True

            # Delete all flows marked with delete_me
            # Add in extracted tables
            new_page = ([], page[1])
            for i, flow in enumerate(page[0]):
                if i in extracted_tables:
                    l = len(new_page[0])-1
                    new_page[0][l:l] = extracted_tables[i]
                if not delete_me[i]:
                    new_page[0].append(flow)

            if first_page:
                first_page = False

            # Filter out abnormally short pages
            page_wc = word_count(new_page)
            if page_wc < medium_limit:
                continue

            yield new_page

        except Exception as e:
            print(f"Error {e} in page\n{page}", file=err_out)

if __name__ == "__main__":
    raise RuntimeError("This file is importable, but not executable")
