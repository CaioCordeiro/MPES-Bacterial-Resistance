# Utils
import sys, os

# Config
from config import Config

# Configuration file

SRA_ID_LIST = ['SRR5386221', 'SRR5386581', 'SRR5387313', 'SRR5386297', 'SRR5386525', 'SRR5385733', 'SRR5385950', 'SRR5386210', 'SRR5385883', 'SRR5386179', 'SRR5386659', 'SRR5386678', 'SRR5385801', 'SRR5386298', 'SRR5385768', 'SRR5385424', 'SRR5386280', 'SRR5386655', 'SRR5385497', 'SRR5387262', 'SRR5386799', 'SRR5386694', 'SRR5386960', 'SRR5386056', 'SRR5386191', 'SRR5385920', 'SRR5385486', 'SRR5385984', 'SRR5386750', 'SRR5385884', 'SRR5385929', 'SRR5385965', 'SRR5385399', 'SRR5386388', 'SRR5386113', 'SRR5386811', 'SRR5385908', 'SRR5387385', 'SRR5386682', 'SRR5386153', 'SRR5387255', 'SRR5385435', 'SRR5386118', 'SRR5385811', 'SRR5387266', 'SRR5386498', 'SRR5385871', 'SRR5386054', 'SRR5385842', 'SRR5386569', 'SRR5386787', 'SRR5385400', 'SRR5387251', 'SRR5386165', 'SRR5386529', 'SRR5386462', 'SRR5386001', 'SRR5385638', 'SRR5385998', 'SRR5385488', 'SRR5385873', 'SRR5385430', 'SRR5386674', 'SRR5385787', 'SRR5385672', 'SRR5386300', 'SRR5387358', 'SRR5386752', 'SRR5385680', 'SRR5385697', 'SRR5385457', 'SRR5387404', 'SRR5385784', 'SRR5386642', 'SRR5386624', 'SRR5385453', 'SRR5385978', 'SRR5385886', 'SRR5386099', 'SRR5386514', 'SRR5386718', 'SRR5386328', 'SRR5386291', 'SRR5386156', 'SRR5386544', 'SRR5386242', 'SRR5385660', 'SRR5385421', 'SRR5386536', 'SRR5386389', 'SRR5386540', 'SRR5385812', 'SRR5386215', 'SRR5387207', 'SRR5386515', 'SRR5387242', 'SRR5387422', 'SRR5385966', 'SRR5385767', 'SRR5386630', 'SRR5387187', 'SRR5386057', 'SRR5386190', 'SRR5385496', 'SRR5385791', 'SRR5386584', 'SRR5386060', 'SRR5386231', 'SRR5387289', 'SRR5386379', 'SRR5385695', 'SRR5386217', 'SRR5387191', 'SRR5387330', 'SRR5387348', 'SRR5386257', 'SRR5385786', 'SRR5385949', 'SRR5386840', 'SRR5387333', 'SRR5385690', 'SRR5386279', 'SRR5387304', 'SRR5386083', 'SRR5386224', 'SRR5386344', 'SRR5386619', 'SRR5386538', 'SRR5387030', 'SRR5386302', 'SRR5386504', 'SRR5386119', 'SRR5386372', 'SRR5386548', 'SRR5385772', 'SRR5387390', 'SRR5386317',
               'SRR5386094', 'SRR5385727', 'SRR5386208', 'SRR5385891', 'SRR5385757', 'SRR5387219', 'SRR5386100', 'SRR5387208', 'SRR5386164', 'SRR5385979', 'SRR5386403', 'SRR5386587', 'SRR5386507', 'SRR5386256', 'SRR5385974', 'SRR5386711', 'SRR5387218', 'SRR5385958', 'SRR5387176', 'SRR5386582', 'SRR5386347', 'SRR5386641', 'SRR5385679', 'SRR5387168', 'SRR5385947', 'SRR5385720', 'SRR5386174', 'SRR5386070', 'SRR5386088', 'SRR5385487', 'SRR5386136', 'SRR5386249', 'SRR5387264', 'SRR5386643', 'SRR5385480', 'SRR5387003', 'SRR5386601', 'SRR5387375', 'SRR5386265', 'SRR5385482', 'SRR5385960', 'SRR5387321', 'SRR5386739', 'SRR5386550', 'SRR5385725', 'SRR5385471', 'SRR5387350', 'SRR5386634', 'SRR5386720', 'SRR5386742', 'SRR5386637', 'SRR5386833', 'SRR5385739', 'SRR5385674', 'SRR5386681', 'SRR5385744', 'SRR5386535', 'SRR5385765', 'SRR5387332', 'SRR5387329', 'SRR5386288', 'SRR5386161', 'SRR5386684', 'SRR5385507', 'SRR5385648', 'SRR5386311', 'SRR5386703', 'SRR5386142', 'SRR5387388', 'SRR5386027', 'SRR5385475', 'SRR5386702', 'SRR5387193', 'SRR5385650', 'SRR5386006', 'SRR5385713', 'SRR5386979', 'SRR5385455', 'SRR5387202', 'SRR5385855', 'SRR5386665', 'SRR5385824', 'SRR5386392', 'SRR5385669', 'SRR5386506', 'SRR5386245', 'SRR5387245', 'SRR5386131', 'SRR5386329', 'SRR5387233', 'SRR5385742', 'SRR5386724', 'SRR5387357', 'SRR5385977', 'SRR5386777', 'SRR5385456', 'SRR5385917', 'SRR5386407', 'SRR5385663', 'SRR5387009', 'SRR5386084', 'SRR5386969', 'SRR5385441', 'SRR5386560', 'SRR5385431', 'SRR5386645', 'SRR5386262', 'SRR5386149', 'SRR5387250', 'SRR5386783', 'SRR5385983', 'SRR5385657', 'SRR5385722', 'SRR5386647', 'SRR5385987', 'SRR5386114', 'SRR5386760', 'SRR5387185', 'SRR5385906', 'SRR5385631', 'SRR5386356', 'SRR5385523', 'SRR5386963', 'SRR5385426', 'SRR5385936', 'SRR5387342', 'SRR5385532', 'SRR5387303', 'SRR5386167', 'SRR5385422', 'SRR5386755', 'SRR5385999', 'SRR5386193', 'SRR5386426', 'SRR5386346', 'SRR5387223', 'SRR5386421', 'SRR5386819']

EXCLUDE_LIST = []
# EXCLUDE_LIST = ['SRR5387035', 'SRR5386005', 'SRR5386522', 'SRR5386967', 'SRR5385681', 'SRR5386345', 'SRR5386075', 'SRR5385436', 'SRR5387211', 'SRR5386629', 'SRR5385677', 'SRR5386934', 'SRR5386319', 'SRR5385470', 'SRR5386237', 'SRR5386778', 'SRR5385864', 'SRR5386192', 'SRR5386843', 'SRR5386572', 'SRR5385477', 'SRR5385967', 'SRR5386431', 'SRR5386683', 'SRR5386276', 'SRR5387413', 'SRR5385641', 'SRR5385914', 'SRR5386255', 'SRR5386680', 'SRR5386181', 'SRR5386043', 'SRR5386424', 'SRR5386132', 'SRR5386456', 'SRR5386032', 'SRR5385749', 'SRR5386834', 'SRR5386950', 'SRR5385469', 'SRR5386448', 'SRR5386500', 'SRR5387227', 'SRR5386821', 'SRR5385735', 'SRR5385872', 'SRR5386625', 'SRR5385723', 'SRR5386430', 'SRR5386848', 'SRR5385769', 'SRR5386413', 'SRR5386490', 'SRR5385930', 'SRR5386466', 'SRR5387254', 'SRR5385449', 'SRR5385425', 'SRR5386495', 'SRR5387361', 'SRR5385968', 'SRR5385658', 'SRR5387244', 'SRR5385654', 'SRR5386390', 'SRR5387389', 'SRR5386613', 'SRR5386744', 'SRR5385932', 'SRR5385662', 'SRR5385445', 'SRR5386066', 'SRR5386229', 'SRR5387386', 'SRR5385623', 'SRR5385434', 'SRR5386143', 'SRR5386338', 'SRR5386241', 'SRR5387025', 'SRR5385796', 'SRR5386417', 'SRR5385452', 'SRR5386211', 'SRR5386815', 'SRR5386452', 'SRR5387288', 'SRR5386578', 'SRR5385524', 'SRR5387396', 'SRR5386145', 'SRR5385489', 'SRR5386203', 'SRR5387001', 'SRR5386026', 'SRR5386926', 'SRR5386489', 'SRR5385633', 'SRR5386212', 'SRR5386552', 'SRR5386180', 'SRR5387032', 'SRR5386134', 'SRR5386436', 'SRR5385454', 'SRR5386814', 'SRR5386129', 'SRR5386793', 'SRR5385693', 'SRR5386411', 'SRR5386713', 'SRR5386465', 'SRR5386830', 'SRR5385636', 'SRR5386628', 'SRR5386588', 'SRR5386638', 'SRR5386205', 'SRR5386767', 'SRR5386571', 'SRR5386393', 'SRR5385923', 'SRR5385905', 'SRR5386322', 'SRR5386771', 'SRR5385806', 'SRR5386475', 'SRR5385646', 'SRR5386077', 'SRR5386439', 'SRR5386441', 'SRR5385462', 'SRR5386374', 'SRR5386454', 'SRR5385635', 'SRR5386772', 'SRR5387395', 'SRR5386101', 'SRR5386334', 'SRR5387230', 'SRR5386437', 'SRR5385972', 'SRR5385691', 'SRR5385533', 'SRR5386616',
#                 'SR   R5386602', 'SRR5386097', 'SRR5387300', 'SRR5387401', 'SRR5386847', 'SRR5386658', 'SRR5386422', 'SRR5385819', 'SRR5386303', 'SRR5385521', 'SRR5385937', 'SRR5385726', 'SRR5385810', 'SRR5387217', 'SRR5386365', 'SRR5385428', 'SRR5386695', 'SRR5386219', 'SRR5385433', 'SRR5387222', 'SRR5387214', 'SRR5386955', 'SRR5387338', 'SRR5386061', 'SRR5386429', 'SRR5386309', 'SRR5386545', 'SRR5386433', 'SRR5385783', 'SRR5385928', 'SRR5386457', 'SRR5386845', 'SRR5386315', 'SRR5385703', 'SRR5387215', 'SRR5386177', 'SRR5386442', 'SRR5387154', 'SRR5386956', 'SRR5385538', 'SRR5386636', 'SRR5386546', 'SRR5386349', 'SRR5386425', 'SRR5386794', 'SRR5385940', 'SRR5386438', 'SRR5385667', 'SRR5387192', 'SRR5385639', 'SRR5386236', 'SRR5386962', 'SRR5387419', 'SRR5385985', 'SRR5386748', 'SRR5387165', 'SRR5385729', 'SRR5386626', 'SRR5385451', 'SRR5385941', 'SRR5386817', 'SRR5387394', 'SRR5386443', 'SRR5386844', 'SRR5387371', 'SRR5386335', 'SRR5385492', 'SRR5387026', 'SRR5385527', 'SRR5386449', 'SRR5386765', 'SRR5385621', 'SRR5386965', 'SRR5385898', 'SRR5387318', 'SRR5385718', 'SRR5387259', 'SRR5386842', 'SRR5386395', 'SRR5385701', 'SRR5385684', 'SRR5386491', 'SRR5386458', 'SRR5386586', 'SRR5385915', 'SRR5385509', 'SRR5385756', 'SRR5385682', 'SRR5385847', 'SRR5387028', 'SRR5385971', 'SRR5387331', 'SRR5386432', 'SRR5385901', 'SRR5387407', 'SRR5387335', 'SRR5386045', 'SRR5386635', 'SRR5386997', 'SRR5386673', 'SRR5386644', 'SRR5386858', 'SRR5387294', 'SRR5386508', 'SRR5387021', 'SRR5386377', 'SRR5385975', 'SRR5387002', 'SRR5386620', 'SRR5385880', 'SRR5386343', 'SRR5387355', 'SRR5386810', 'SRR5387277', 'SRR5385897', 'SRR5386002', 'SRR5386944', 'SRR5385835', 'SRR5386086', 'SRR5387414', 'SRR5387427', 'SRR5386808', 'SRR5386423', 'SRR5385797', 'SRR5386049', 'SRR5386929', 'SRR5386275', 'SRR5386050', 'SRR5386719', 'SRR5385466', 'SRR5386206', 'SRR5386071', 'SRR5386743', 'SRR5386418', 'SRR5385826', 'SRR5385833', 'SRR5386472', 'SRR5386615', 'SRR5385926', 'SRR5385736', 'SRR5386676', 'SRR5386478', 'SRR5386283', 'SRR5386025', 'SRR5386046', 'SRR5386534']

ANTIBIOTIC_LIST = ['Amikacin', 'Ampicillin', 'Ampicillin/Sulbactam', 'Aztreonam', 'Cefazolin', 'Cefepime', 'Cefoxitin', 'Ceftazidime', 'Ceftriaxone', 'Cefuroxime sodium',
                   'Ciprofloxacin', 'Gentamicin', 'Imipenem', 'Levofloxacin', 'Meropenem', 'Nitrofurantoin', 'Piperacillin/Tazobactam', 'Tetracycline', 'Tobramycin', 'Trimethoprim/Sulfamethoxazole']

list(set(SRA_ID_LIST) - set(EXCLUDE_LIST))



def download_file(experiment_id):
    url = "https://trace.ncbi.nlm.nih.gov/Traces/sra/sra.cgi?cmd=dload&run_list={}&format=fasta".format(
        experiment_id)

    try:
        os.system(
            "fasterq-dump {} -O data/raw_data/{}".format(experiment_id, experiment_id))
    except Exception as error:
        print('Error downloading file for: ', experiment_id, "\n", error)


def main():
    experiment_counter = 0
    download_counter = 0

    for experiment_id in SRA_ID_LIST:
        download_counter += 1

        print("Downloading file: {}/{}".format(download_counter,
              len(SRA_ID_LIST)), end="\r")
        download_file(experiment_id)


if __name__ == "__main__":
    main()
