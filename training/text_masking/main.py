from generator import SampleGenerator, Locale, GeneratorConfig, Background

english_config = GeneratorConfig(
    locales=[
        Locale("en_US"),
        Locale("en_GB"),
    ],
    font_files=[
        "fonts/animeace2_bld.ttf",
        "fonts/animeace2_ital.ttf",
        "fonts/BlambotClassicBB-Bold.ttf",
        "fonts/BlambotClassicBB-Italic.ttf",
        "fonts/BlambotClassicBB.ttf",
        "fonts/Roboto-Light.ttf",
        "fonts/Roboto-Regular.ttf",
        "fonts/Roboto-Bold.ttf",
        "fonts/Roboto-Italic.ttf",
    ],
)

japanese_config = GeneratorConfig(
    locales=[
        Locale("ja", ""),
    ],
    font_files=[
        "fonts/07YasashisaAntique.otf",
        "fonts/07YasashisaBold.ttf",
        "fonts/kowaiFont.ttf",
        "fonts/LanobePOPv2.otf",
        "fonts/M-NijimiMincho.otf",
        "fonts/msmincho.ttf",
        "fonts/NikumaruFont.otf",
        "fonts/reiko.ttf",
        "fonts/YasashisaGothicBold-V2.otf",
        "fonts/SourceHanSans.ttc",
    ],
)

mixed_config = GeneratorConfig(
    locales=[
        Locale("ko", sep=""),
        Locale("zh_CN", sep=""),
        Locale("zh_TW", sep=""),
    ],
    font_files=["fonts/SourceHanSans.ttc"],
)


def main():
    gen = SampleGenerator(
        [english_config, japanese_config, mixed_config],
        [
            Background.file("bg/132693212_p0_master1200.jpg"),
            Background.file("bg/141664265_p0_master1200.jpg"),
            Background.file("bg/142016801_p0_master1200.jpg"),
            Background.file("bg/142108984_p0_master1200.jpg"),
            Background.file("bg/142134010_p0_master1200.jpg"),
            Background.file("bg/142154762_p0_master1200.jpg"),
            Background.file("bg/142222754_p0_master1200.jpg"),
            Background.file("bg/142288643_p0_master1200.jpg"),
            Background.file("bg/142292178_p0_master1200.jpg"),
            Background.file("bg/142302973_p0_master1200.jpg"),
        ],
    )

    gen.run("./seg_dataset/train", count=10000, seed=0)
    gen.run("./seg_dataset/valid", count=40, seed=30)
    gen.run("./seg_dataset/test", count=40, seed=65)

    # gen.run("./debug", count=50)


if __name__ == "__main__":
    main()
