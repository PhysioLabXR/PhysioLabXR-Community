import os

from openpyxl import Workbook


def create_study_sessions():
    """
    Creates 50 Excel session files named session<#>_study_3.xlsx.

    Each session has 14 sentences:
      - The next 10 lines from mackenzie_soukoreff_phrase_set.txt
      - The next 4 lines from eronmobile_email_dataset_mem.txt

    This will use all 700 lines (500 + 200) exactly once across the 50 files.
    """
    # Read the lines from the two text files
    with open('mackenzie_soukoreff_phrase_set.txt', 'r', encoding='utf-8') as f_mack:
        mackenzie_lines = [line.strip() for line in f_mack.readlines()]

    with open('eronmobile_email_dataset_mem.txt', 'r', encoding='utf-8') as f_eron:
        eronmobile_lines = [line.strip() for line in f_eron.readlines()]

    # Keep track of our position in each file
    m_index = 0  # For Mackenzie–Soukoreff file
    e_index = 0  # For EronMobile file

    # Generate 50 session files
    for session_num in range(1, 51):
        # Create a new workbook
        wb = Workbook()
        ws = wb.active

        # Pull out the next 10 from Mackenzie–Soukoreff
        session_mackenzie = mackenzie_lines[m_index: m_index + 10]
        m_index += 10

        # Pull out the next 4 from EronMobile
        session_eronmobile = eronmobile_lines[e_index: e_index + 4]
        e_index += 4

        # Combine them to form the 14 trials for this session
        session_lines = session_mackenzie + session_eronmobile

        # Write each line to a row in the spreadsheet
        for row_idx, line in enumerate(session_lines, start=1):
            ws.cell(row=row_idx, column=1).value = line

        # Save the file: session1_study_3.xlsx, session2_study_3.xlsx, etc.
        filename = f"session{session_num}_study_3.xlsx"
        wb.save(os.path.join(export_dir, filename))
        print(f"Created {filename}")



export_dir = '../StudySentences/user_study3_session_sentences'

if __name__ == "__main__":
    create_study_sessions()