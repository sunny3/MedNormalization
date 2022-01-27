class MeddraHelper():
    def __init__(self, meddra_path):
        self.meddra_path = meddra_path
        self.ptcodes = []
        self.ptcode_to_pt = {}
        self.llt_to_lltcode = {}
        self.lltcode_to_ptcode = {}
        self.Load()

    def Load(self):
        """
        собираем из медры только то, что надо, а именно:
        - все пт коды для быстрой проверки, является ли найденный где-то код PT кодом
        - словарь перевода из ллт кода в соответствующий пт код
        - словарь перевода llt в llt код
        """
        with open(self.meddra_path + "pt.asc", "r") as f:
            for pt_line in f:
                pt_line = pt_line.split("$")
                self.ptcodes.append(pt_line[0])
                self.ptcode_to_pt[pt_line[0]] = pt_line[1]
        with open(self.meddra_path + "/llt.asc", "r") as f:
            for llt_line in f:
                llt_line = llt_line.split("$")
                self.lltcode_to_ptcode[llt_line[0]] = llt_line[2]
                self.llt_to_lltcode[llt_line[1]] = llt_line[0]