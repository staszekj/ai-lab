"""New Testament book definitions with TextGrid aggregation IDs."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Book:
    name: str
    name_pl: str
    code: str
    textgrid_id: str

    @property
    def url(self) -> str:
        return f"https://textgridlab.org/1.0/tgcrud-public/rest/textgrid:{self.textgrid_id}/data"


NT_BOOKS: list[Book] = [
    Book("Matthew", "Ewangelia wg św. Mateusza", "MAT", "49b3t.0"),
    Book("Mark", "Ewangelia wg św. Marka", "MRK", "49b3x.0"),
    Book("Luke", "Ewangelia wg św. Łukasza", "LUK", "49b41.0"),
    Book("John", "Ewangelia wg św. Jana", "JHN", "49b44.0"),
    Book("Acts", "Dzieje Apostolskie", "ACT", "49b47.0"),
    Book("Romans", "List do Rzymian", "ROM", "49b4b.0"),
    Book("1 Corinthians", "1 List do Koryntian", "1CO", "49b4f.0"),
    Book("2 Corinthians", "2 List do Koryntian", "2CO", "49b4j.0"),
    Book("Galatians", "List do Galacjan", "GAL", "49b4n.0"),
    Book("Ephesians", "List do Efezjan", "EPH", "49b4r.0"),
    Book("Philippians", "List do Filipian", "PHP", "49b4v.0"),
    Book("Colossians", "List do Kolosan", "COL", "49b4z.0"),
    Book("1 Thessalonians", "1 List do Tesaloniczan", "1TH", "49b52.0"),
    Book("2 Thessalonians", "2 List do Tesaloniczan", "2TH", "49b55.0"),
    Book("1 Timothy", "1 List do Tymoteusza", "1TI", "49b58.0"),
    Book("2 Timothy", "2 List do Tymoteusza", "2TI", "49b5c.0"),
    Book("Titus", "List do Tytusa", "TIT", "49b5g.0"),
    Book("Philemon", "List do Filemona", "PHM", "49b5k.0"),
    Book("Hebrews", "List do Hebrajczyków", "HEB", "49b5p.0"),
    Book("James", "List św. Jakuba", "JAS", "49b5s.0"),
    Book("1 Peter", "1 List św. Piotra", "1PE", "49b5w.0"),
    Book("2 Peter", "2 List św. Piotra", "2PE", "49b60.0"),
    Book("1 John", "1 List św. Jana", "1JN", "49b63.0"),
    Book("2 John", "2 List św. Jana", "2JN", "49b66.0"),
    Book("3 John", "3 List św. Jana", "3JN", "49b69.0"),
    Book("Jude", "List św. Judy", "JUD", "49b6d.0"),
    Book("Revelation", "Apokalipsa św. Jana", "REV", "49b6h.0"),
]
